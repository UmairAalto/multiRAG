from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import logging
import os
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from httpx import Client, Request
import uuid
import base64
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import numpy as np

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


# Create a logger for this module
logger = logging.getLogger(__name__)


# Load environment variables (if needed)
load_dotenv()

# API keys and URLs from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Function to process PDF and split it into chunks
def process_pdf(pdf_path):
    """Process the PDF, split it into chunks, and return the chunks."""
    print(pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    document_text = "".join([page.page_content for page in pages])

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=200  # Adjust as needed
    )
    chunks = text_splitter.create_documents([document_text])

    return chunks

def chunk_pages(pages, chunk_size, overlap):
    """
    Combines page contents, splits the combined text into chunks, includes metadata
    and returns the chunks
    
    Args:
        contents (list): List of content items (with page_idx and type).
        chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of overlapping characters between consecutive chunks.
        
    Returns:
        list: A list of dictionaries, each with keys:
              - "text": The chunk text.
              - "pages": A string describing the page(s) the chunk covers.
    """
    
    # Sort the pages by page number and build a single combined text.
    combined_text = ""
    boundaries = []  # list of tuples: (page_number, start_index, end_index)
    
    # combine page contents
    for page in sorted(pages.keys()):
        start = len(combined_text)
        page_text = pages[page].strip()
        combined_text += page_text
        end = len(combined_text)
        boundaries.append((page, start, end))
    
    
    # Chunking the combined text
    chunks = []
    step = chunk_size - overlap
    chunk_start = 0
    text_length = len(combined_text)
    
    while chunk_start < text_length:
        chunk_end = min(chunk_start + chunk_size, text_length)
        chunk_text = combined_text[chunk_start:chunk_end]
        
        # Page numbering
        pages_in_chunk = []
        for page, start, end in boundaries:

            if end > chunk_start and start < chunk_end:
                pages_in_chunk.append(page)
        
        if pages_in_chunk:
            pages_in_chunk.sort()
            
            if len(pages_in_chunk) == 1:
                # If chunk is in one page
                page_meta = str(pages_in_chunk[0] + 1)
            else:
                # If a chunk overlaps with multiple pages
                page_meta = f"{pages_in_chunk[0] + 1}-{pages_in_chunk[-1] + 1}"
        
        
        chunks.append({"text": chunk_text, "pages": page_meta})
        
        # Move the window forward.
        chunk_start += step
    
    return chunks

def combine_page_contents(contents):
    
    pages = {}
    
    for item in contents:
        page = item.get("page_idx")
        if page not in pages:
            pages[page] = []
        
        if item["type"] == "table":
            # Join the table caption (if available) and the table body.
            caption = "".join(item.get("table_caption", [])).strip()
            table_body = item.get("table_body", "").strip()
            # Combine caption and table body with proper spacing/newlines.
            combined = f"{caption}   \n{table_body}"

        elif item["type"] == "image":
            # Format image using markdown and append its caption.
            caption = "".join(item.get("img_caption", [])).strip()
            combined = f"![](Here is an image of the figure)  \n{caption}"

        elif item["type"] == "text":
            text = item.get("text", "").strip()
            # If text_level is provided, prefix the text with that many '#' characters.
            if "text_level" in item:
                level = item["text_level"]
                text = f"{'#' * level} {text}"
            combined = text
        elif item["type"] == "equation":
            combined = item.get("text", "").strip()
        else:
            # If the type is not recognized, skip this element.
            continue
        
        pages[page].append(combined)
    
    # Join the individual parts for each page using two newlines.
    for page in pages:
        pages[page] = "\n\n".join(pages[page])
    
    return pages

def combine_vectors_avg(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return (0.5 * vec1 + 0.5 * vec2)

def get_images(md_contents, local_dir, img_embedding_model):
    
    images = []

    for item in md_contents:
        if item["type"] == "image":
            with open("output/"+item["img_path"], "rb") as img_file:
                file_bytes = img_file.read()
                base64_str = base64.b64encode(file_bytes).decode("utf-8")
                img_embd = img_embedding_model.embed_image([local_dir + item["img_path"]])
                
                if item.get("img_caption"):
                    caption = item["img_caption"][0]
                    txt_embd = img_embedding_model.embed_query(item["img_caption"][0])
                    embd = combine_vectors_avg(img_embd[0], txt_embd)
                else:
                    caption = "This figure doesn't have a caption"
                    embd = img_embd[0]

                images.append({"base64": base64_str, 
                            "img_embeddings": embd,
                            "page_num": item["page_idx"] + 1,
                            "image_caption": caption})
    return images


def process_pdf_with_tables(pdf_path, img_embedding_model):
    # args
    pdf_file_name = pdf_path  # replace with the real pdf path
    
    # prepare env
    local_image_dir, local_md_dir = "output/images", "output"
    image_dir = str(os.path.basename(local_image_dir))

    os.makedirs(local_image_dir, exist_ok=True)

    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )

    # read bytes
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

    # proc
    ## Create Dataset Instance
    ds = PymuDocDataset(pdf_bytes)

    ## inference

    infer_result = ds.apply(doc_analyze, ocr=True)

    ## pipeline
    pipe_result = infer_result.pipe_ocr_mode(image_writer)

    ### get content list content
    content_list_content = pipe_result.get_content_list(image_dir)

    pages = combine_page_contents(content_list_content)

    chunks = chunk_pages(pages, 1000, 200) # Adjust as needed

    images = get_images(content_list_content, f"{local_md_dir}/", img_embedding_model)

    return chunks, images

# Function to send document chunks (with embeddings) to the Qdrant vector database
def send_to_qdrant(filename, documents, images, txt_embedding_model):
    """Send the document chunks to the Qdrant vector database."""
    try:
        
        client = QdrantClient(url=QDRANT_URL)
        collection = "test_image" # Replace with your collection name

        if not client.collection_exists(collection_name=collection):
            client.create_collection(
                collection_name=collection,
                vectors_config={
                    "image": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=3072, distance=models.Distance.COSINE),
                }
            )
        
        if len(images) > 0:

            client.upload_points(
                collection_name=collection,
                points=[
            models.PointStruct(
                        id=str(uuid.uuid4()), #unique id of a point
                        vector={
                            "image": image["img_embeddings"], #embeded image
            },
                        payload={"page_content": image["base64"],
                                "metadata": {"image_caption": image["image_caption"], "filename": filename,"Page": image["page_num"]}} #original image and its caption
            )
                    for image in images
            ]
            )
        
        client.upload_points(
                collection_name=collection,
                points=[
            models.PointStruct(
                        id=str(uuid.uuid4()), #unique id of a point
                        vector={
                            "text": txt_embedding_model.embed_query(chunk["text"]) #embeded text chunk
            },
                        payload={"page_content": chunk["text"],
                                "metadata": {"filename": filename, "Page(s)": chunk["pages"]}} #original text chunk
            )
                    for chunk in documents
            ]
            )

        return True
    except Exception as ex:
        print(f"Failed to store data in the vector DB: {str(ex)}")
        return False
    
# Function to initialize the Qdrant client and return the vector store object
def qdrant_client(txt_embedding_model, image_embedding_model):
    """Initialize Qdrant client and return the vector store."""
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    collection = "test_image"
    txt_qdrant_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection,
        embedding=txt_embedding_model,
        vector_name="text"
    )
    img_qdrant_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection,
        embedding=image_embedding_model,
        vector_name="image"
    )
    
    return txt_qdrant_store, img_qdrant_store


# Function to handle question answering using the Qdrant vector store and GPT
def qa_ret(text_store, image_store, input_query):
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        #langchain.debug = True
        
        txt_retriever = text_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        img_retriever = image_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4, "score_threshold": 0.8}
        )

        messages = [
            ("system", """Instructions:
            You are an expert compliance analyst specializing in the maritime industry. Your task is to extract precise answers using the provided Context (text chunks from maritime standards), Images (figures from the standards), and the User’s Question. Your response must be based on a semantic understanding of the content.
            
            **Note:** If the Context references a figure, the corresponding image will be uploaded along with its caption.
             
            **Key Guidelines:**
            - **Answer Length**: Provide an answer between 40 and 100 words.
            - **Conciseness & Focus**: Include only the necessary information to directly address the question.
            - **Professional Tone**: Use polite, formal language and avoid any abusive or prohibited expressions.
            - **Privacy**: Do not request personal information.
            - **Semantic Inference**: If exact wording is unavailable, infer the closest meaning using natural language understanding.
            - **Unavailable Information**: If the needed information is not present in the Context, politely apologize and state that it is not available.
            - **Response Format**: Use markdown formatting for headings, lists, and mathematical expressions. For all mathematical expressions, use LaTeX enclosed in double dollar signs for display math (e.g.,$$a=b \\cdot c$$) and single dollar signs for inline math (e.g., $t_0$). Do not use any upgreek commands (e.g., avoid \\uprho). Instead, use standard LaTeX commands for Greek letters (e.g., \\rho). Do not use square brackets to delimit formulas.
            - **Images Integration**: Evaluate images as supplementary context and include relevant interpretations if needed. When the Context references a figure, use the uploaded image and its caption to support your answer.
            - **Traceability**: If your answer is directly derived from the provided Context, append a reference to the specific page(s) and file name(s) from which the information was extracted (e.g., "Source: [DocumentName.pdf, Page 3]"). For multiple sources, separate each reference accordingly. Do not include reference(s) if your answer is not based on the Context.
            
            Respond in a polite, professional, and concise manner."""),
            ("human", "Context: {context}"),
            ("human", "**User's Question:** {question}")
        ]

        images = img_retriever.invoke(input_query)

        img = False
        if len(images) > 0: 
            img = True
            content = []
            for image in images:
                content.append({"type": "text", 
                "text": f"{image.metadata['image_caption']}"})
                content.append(
                {'type': 'image_url', 
                'image_url': {'url': f'data:image/png;base64,{image.page_content}'}})
            messages.append(("human", content))
        
        # Langfuse callback
        #user_id = f"qdrant"
        #langfuse_handler = get_callback_handler(user_id)

        prompt = ChatPromptTemplate.from_messages(messages)

        setup_and_retrieval = RunnableParallel(
            {"context": txt_retriever, "question": RunnablePassthrough()}
        )#.with_config({"callbacks": [langfuse_handler]})

        # Get LLM model
        model = get_llm_model()

        output_parser = StrOutputParser()
        
        rag_chain = setup_and_retrieval | prompt | model | output_parser

        response = rag_chain.invoke(input_query)
        
        if img:
            return response, images
        return response, ""

    except Exception as ex:
        return f"Error: {str(ex)}"


# Function that return langfuse callback handler
def get_callback_handler(username):

    try:
        logger.debug(f"Landfuse: getting public key {os.environ['LANGFUSE_PUBLIC_KEY']} and host: {os.environ['LANGFUSE_HOST']}")
        langfuse_handler = CallbackHandler(
            #user_id=username,
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
            )
        langfuse_handler.auth_check()
        return langfuse_handler
    except KeyError as e:
        logger.error(f"Environment variable {e} not found")
        return ""
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return ""


# Get embedding_model
def get_embedding_models():
    
    print("Use Azure OpenAI API")
    # Create the embedding model for Azure OpenAI
    embedding_model = AzureOpenAIEmbeddings(
        api_version = "2024-02-01",
        model="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        default_headers={"Ocp-Apim-Subscription-Key": os.getenv("AZURE_OPENAI_API_KEY")},

    )

    clip = OpenCLIPEmbeddings(model_name='ViT-H-14-378-quickgelu', checkpoint='dfn5b')

    return embedding_model, clip

def update_base_url(request: Request) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path="/v1/openai/deployments/gpt-4o-2024-08-06/chat/completions")

# Get llm model
def get_llm_model():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    # Create the embedding model for Azure OpenAI

    """model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        gemini_api_key=gemini_api_key
    )"""
    model = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        base_url="https://aalto-openai-apigw.azure-api.net/",
        default_headers={"Ocp-Apim-Subscription-Key": api_key},
        http_client=Client(event_hooks={"request": [update_base_url]})
    )

    return model
