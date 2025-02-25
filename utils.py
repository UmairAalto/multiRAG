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
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
from langchain_experimental.open_clip import OpenCLIPEmbeddings

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

def get_images(md_contents, local_dir):
    
    images = []

    clip_embd = OpenCLIPEmbeddings(model_name='ViT-H-14-378-quickgelu', checkpoint='dfn5b')

    for item in md_contents:
        if item["type"] == "image":
            embd = clip_embd.embed_image([local_dir + item["img_path"]])
            
            #encoded_bytes = base64.b64encode(file_bytes).decode("utf-8")
            image_url = f"http://localhost:8000/{item['img_path']}"
            #print(len(embd[0]))
            images.append({"url": image_url, 
                        "embeddings": embd[0],
                        "page_num": item["page_idx"] + 1,
                        #"filename": file1,
                        "image_caption": item["img_caption"][0]})
    return images


def process_pdf_with_tables(pdf_path):
    # args
    pdf_file_name = pdf_path  # replace with the real pdf path
    name_without_suff = pdf_file_name.split(".")[0]
    
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


    ### get markdown content
    md_content = pipe_result.get_markdown(image_dir)

    ### dump markdown
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    ### get content list content
    content_list_content = pipe_result.get_content_list(image_dir)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=200,  # Adjust as needed
        separators=[""]
    )
    chunks = text_splitter.create_documents([md_content])

    images = get_images(content_list_content, f"{local_md_dir}/")

    return chunks, images

# Function to send document chunks (with embeddings) to the Qdrant vector database
def send_to_qdrant(documents, images, embedding_model):
    """Send the document chunks to the Qdrant vector database."""
    try:
        """qdrant = QdrantVectorStore.from_documents(
            documents,
            embedding_model,
            url=QDRANT_URL,
            prefer_grpc=False,
            #api_key=QDRANT_API_KEY,
            collection_name="xeven_chatbot",  # Replace with your collection name
            force_recreate=True  # Create a fresh collection every time
        )"""
        client = QdrantClient(url=QDRANT_URL)
        collection = "text_image" # Replace with your collection name

        if not client.collection_exists("text_image"):
            client.create_collection(
                collection_name=collection,
                vectors_config={
                    "image": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=3072, distance=models.Distance.COSINE),
                }
            )
        print(f"image length: {len(images)}\n")
        if len(images) > 0:

            client.upload_points(
                collection_name=collection,
                points=[
            models.PointStruct(
                        id=str(uuid.uuid4()), #unique id of a point, pre-defined by the user
                        vector={
                            "image": image["embeddings"] #embeded image
            },
                        payload={"page_content": image["url"],
                                "metadata": {"image_caption": image["image_caption"], "page_num": image["page_num"]}} #original image and its caption
            )
                    for image in images
            ]
            )
        
        client.upload_points(
                collection_name=collection,
                points=[
            models.PointStruct(
                        id=str(uuid.uuid4()), #unique id of a point, pre-defined by the user
                        vector={
                            "text": embedding_model.embed_query(chunk.page_content) #embeded text chunk
            },
                        payload={"page_content": chunk.page_content} #original text chunk
            )
                    for chunk in documents
            ]
            )

        return True
    except Exception as ex:
        print(f"Failed to store data in the vector DB: {str(ex)}")
        return False

class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model
        
    def embed_query(self, text: str) -> list:
            # Ensure the model returns a numpy array and flatten it
            emb = self.model.encode(text, convert_to_numpy=True).flatten()
            # Convert each element to a native Python float
            return [float(x) for x in emb]

    def embed_documents(self, texts: list) -> list:
        # For a list of texts, process each individually
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Ensure the output is 2D (each row is an embedding)
        return [[float(x) for x in emb] for emb in embeddings]
    
# Function to initialize the Qdrant client and return the vector store object
def qdrant_client():
    """Initialize Qdrant client and return the vector store."""
    print("Use Azure OpenAI API")
    # Create the embedding model for Azure OpenAI
    embedding_model = AzureOpenAIEmbeddings(
        api_version = "2024-02-01",
        model="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        default_headers={"Ocp-Apim-Subscription-Key": os.getenv("AZURE_OPENAI_API_KEY")},
    )
    
    #img_embedding_model = SentenceTransformer("clip-ViT-B-32")
    #wrapped_img_model = SentenceTransformerWrapper(img_embedding_model)
    clip_embd = OpenCLIPEmbeddings(model_name='ViT-H-14-378-quickgelu', checkpoint='dfn5b')

    qdrant_client = QdrantClient(url=QDRANT_URL)
    #qdrant_client.create_collection()
    qdrant_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="text_image",
        embedding=clip_embd,
        vector_name="image"
    )
    
    return qdrant_store


# Function to handle question answering using the Qdrant vector store and GPT
def qa_ret(qdrant_store, input_query):
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        template = """
        Instructions:
            You are trained to extract answers from the given Context and the User's Question. Your response must be based on semantic understanding, which means even if the wording is not an exact match, infer the closest possible meaning from the Context. 

            Key Points to Follow:
            - **Precise Answer Length**: The answer must be between a minimum of 40 words and a maximum of 100 words.
            - **Strict Answering Rules**: Do not include any unnecessary text. The answer should be concise and focused directly on the question.
            - **Professional Language**: Do not use any abusive or prohibited language. Always respond in a polite and gentle tone.
            - **No Personal Information Requests**: Do not ask for personal information from the user at any point.
            - **Concise & Understandable**: Provide the most concise, clear, and understandable answer possible.
            - **Semantic Similarity**: If exact wording isn’t available in the Context, use your semantic understanding to infer the answer. If there are semantically related phrases, use them to generate a precise response. Use natural language understanding to interpret closely related words or concepts.
            - **Unavailable Information**: If the answer is genuinely not found in the Context, politely apologize and inform the user that the specific information is not available in the provided context.

            Context:
            {context}

            **User's Question:** {question}

            Respond in a polite, professional, and concise manner.
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = qdrant_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        contxt = retriever.invoke(input_query)
        print(f"Retrieved context: {contxt}\n")
        # Langfuse callback
        user_id = f"qdrant"
        #langfuse_handler = get_callback_handler(user_id)

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )#.with_config({"callbacks": [langfuse_handler]})

        # Get LLM model
        model = get_llm_model()

        output_parser = StrOutputParser()

        rag_chain = setup_and_retrieval | prompt | model | output_parser
        
        response = rag_chain.invoke(input_query)
        
        return response

    except Exception as ex:
        return f"Error: {str(ex)}"


# Function that return langfuse callback handler
def get_callback_handler(username):

    try:
        logger.debug(f"Landfuse: getting public key {os.environ['LANGFUSE_PUBLIC_KEY']} and host: {os.environ['LANGFUSE_HOST']}")
        langfuse_handler = CallbackHandler(
            user_id=username,
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
def get_embedding_model():
    
    print("Use Azure OpenAI API")
    # Create the embedding model for Azure OpenAI
    embedding_model = AzureOpenAIEmbeddings(
        api_version = "2024-02-01",
        model="text-embedding-3-large",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        default_headers={"Ocp-Apim-Subscription-Key": os.getenv("AZURE_OPENAI_API_KEY")},

    )

    return embedding_model

def update_base_url(request: Request) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path="/v1/openai/deployments/gpt-4o-2024-08-06/chat/completions")

# Get llm model
def get_llm_model():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Create the embedding model for Azure OpenAI
    model = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        base_url="https://aalto-openai-apigw.azure-api.net/",
        default_headers={"Ocp-Apim-Subscription-Key": api_key},
        http_client=Client(event_hooks={"request": [update_base_url]})
    )

    return model
