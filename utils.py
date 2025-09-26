from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import logging
import os
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from httpx import Client, Request
import uuid
import base64
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import numpy as np

import copy
import json
import os
from pathlib import Path

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


# Create a logger for this module
logger = logging.getLogger(__name__)


# Load environment variables (if needed)
load_dotenv()

# API keys and URLs from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def overlap_text_splitter(contents, chunk_size=1500, overlap=250):
    tmp_chunk_size = chunk_size
    step = chunk_size - overlap
    tmp_step = step
    chunks = []
    chunk_text = ""
    page_idx = []
    for item in contents:
        page = item.get("page_idx")
        
        if page not in page_idx:
            page_idx.append(page)
        
        if item["type"] == "text":
            text = item.get("text").strip()
            # If text_level is provided, prefix the text with that many '#' characters.
        
            if "text_level" in item:
                level = item["text_level"]
                text = f"{'#' * level} {text}"
            
            text += "\n\n"

        elif item["type"] == "table":
            # Join the table caption (if available) and the table body.
            caption = "".join(item.get("table_caption", [])).strip()
            table_body = item.get("table_body", "").strip()
            # Combine caption and table body with proper spacing/newlines.
            text = f"{caption}   \n{table_body}\n\n"
            if len(text) > tmp_chunk_size:
                chunk_text += text
                if len(page_idx) > 1:

                    chunks.append({"page_idx": f"{page_idx[0]}-{page_idx[-1]}", "text": chunk_text})
                else:
                    chunks.append({"page_idx": f"{page_idx[0]}", "text": chunk_text})
                
                chunk_text = ""
                page_idx.clear()
                tmp_chunk_size = chunk_size
                continue


        elif item["type"] == "image":
            # Format image using markdown and append its caption.
            caption = "".join(item.get("img_caption", [])).strip()
            text = f"![](Here is an image of the figure)  \n{caption}\n\n"

        elif item["type"] == "equation":
            text = item.get("text", "").strip()
            text += "\n\n"
        else:
            # If the type is not recognized, skip this element.
            continue
        

        if len(text) <= tmp_chunk_size:
            chunk_text += text
            tmp_chunk_size -= len(text)
            tmp_step -= len(text)

        
        else:
            chunk_text += text[:tmp_chunk_size]
            text = text[tmp_chunk_size:]

            if len(page_idx) > 1:

                chunks.append({"page_idx": f"{page_idx[0]}-{page_idx[-1]}", "text": chunk_text})
            else:
                chunks.append({"page_idx": f"{page_idx[0]}", "text": chunk_text})
            
            chunk_text = chunk_text[-overlap:]

            while len(text) > step:
                #chunk_text = text[:chunk_size]
                chunk_text += text[:step]
                chunks.append({"page_idx": f"{page_idx[-1]}", "text": chunk_text})
                chunk_text = chunk_text[-overlap:]
                text = text[step:]
            
            tmp_chunk_size = step
            if len(text) < step:
                chunk_text += text
                tmp_chunk_size -= len(text)
            page_idx.clear()
        
    if chunk_text:
        num = chunks[-1]["page_idx"]
        if "-" in num:
            num.split("-")
            chunks.append({"page_idx": num[2], "text": chunk_text})
        else:
            chunks.append({"page_idx": num, "text": chunk_text})

    return chunks
# Function to process PDF and split it into chunks
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
                else:
                    caption = "This figure doesn't have a caption"

                images.append({"base64": base64_str, 
                            "img_embeddings": img_embd[0],
                            "page_num": item["page_idx"] + 1,
                            "image_caption": caption})
    return images

def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=False,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=False,  # Whether to dump middle JSON files
    f_dump_model_output=False,  # Whether to dump model output files
    f_dump_orig_pdf=False,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable)

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url)

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")

def process_pdf_with_tables(path_list: list[Path],
        output_dir,
        lang="en",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None):
    try:
        """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
        
        return True
    except Exception as ex:
        print(f"Failed to extract pdf: {str(ex)}")
        return False

# Function to send document chunks (with embeddings) to the Qdrant vector database
def send_to_qdrant(filename, documents, images, txt_embedding_model, collection="All"):
    """Send the document chunks to the Qdrant vector database."""
    try:
        
        client = QdrantClient(url=QDRANT_URL)

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
def qdrant_client(txt_embedding_model, image_embedding_model, collection="All"):
    """Initialize Qdrant client and return the vector store."""
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
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
def qa_ret(text_store, image_store, input_query, k=4):
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        
        txt_retriever = text_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        img_retriever = image_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3, "score_threshold": 0.359}
        )

        messages = [
            ("system", """Instructions:
            You are an expert compliance analyst specializing in the maritime industry. Your task is to extract precise answers using the provided Context (text chunks from maritime standards), Images (figures from the standards), and the User’s Question. Your response must be based on a semantic understanding of the content.
            
            **Note:** If the Context references a figure, the corresponding image will be uploaded along with its caption.
             
            **Key Guidelines:**
            - **Answer Length**: Provide an answer between 40 and 300 words.
            - **Conciseness & Focus**: Include only the necessary information to directly address the question.
            - **Professional Tone**: Use polite, formal language and avoid any abusive or prohibited expressions.
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
        user_id = f"qdrant"
        langfuse_handler = get_callback_handler(user_id)

        prompt = ChatPromptTemplate.from_messages(messages)

        setup_and_retrieval = RunnableParallel(
            {"context": txt_retriever, "question": RunnablePassthrough()}
        )

        # Get LLM model
        model = get_llm_model()

        output_parser = StrOutputParser()
        
        rag_chain = setup_and_retrieval | prompt | model | output_parser

        response = rag_chain.invoke(input_query, config={"callbacks": [langfuse_handler]})
        
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
            user_id=username,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
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
        model=os.getenv("EMBEDDING"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    clip = OpenCLIPEmbeddings(model_name='ViT-H-14-378-quickgelu', checkpoint='dfn5b')

    return embedding_model, clip

def update_base_url(request: Request) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path="/v1/openai/deployments/gpt-4o-2024-08-06/chat/completions")


# Get llm model
def get_llm_model():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    # Create the embedding model for Azure OpenAI
    
    model = AzureChatOpenAI(
        temperature=0,
        api_key=api_key,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("LLM"),
        api_version="2024-12-01-preview"
    )

    return model
