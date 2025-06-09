from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
from langchain.prompts import ChatPromptTemplate
from typing import Any
import logging
import torch
from contextlib import asynccontextmanager
from dotenv import load_dotenv, find_dotenv
import json

# Import the necessary functions from utils.py
from utils import process_pdf, send_to_qdrant, qdrant_client, qa_ret, combine_page_contents, get_embedding_models, process_pdf_with_tables, chunk_pages, get_images


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger  = logging.getLogger('backend')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize your embedding models once on startup.
    txt_model, clip_model = get_embedding_models()
    app.state.embedding_models = {"txt": txt_model, "clip": clip_model}
    print("Embedding models loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)
# Keep tracks of user specific variables, set in login function
session = {} # CHANGED TO SE flask_session instead, extention to handle server side sessions
# Hold agents for users,
# Key-value pairs are the username and AgentExecutor object with the user specific agent
agents = {}

# Frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Loading environment variables
load_dotenv(find_dotenv())


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[LANGFUSE_HOST, FRONTEND_URL],  # Allow requests from your React app (adjust domain if necessary)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)



class QuestionRequest(BaseModel):
    question: str
    n_chunks: int
    collection: str

class SendFilesRequest(BaseModel):
    filename: str
    collection: str


# Define a prompt for the API
class RAGChatPromptTemplate(ChatPromptTemplate):
    prompt:str
    
    def __init__(self, template:str):
        self.prompt = ChatPromptTemplate.from_template(template)


# Endpoint to upload a PDF and process it, sending to Qdrant
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file, extract data and save it.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            filename = file.filename[:-4]
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
            
        
        # Process the PDF to get document chunks and embeddings
        success = process_pdf_with_tables(temp_file_path, filename)
        
        # Remove the temporary file after processing
        os.remove(temp_file_path)

        if success:
            return {"message": "PDF successfully processed and stored in vector DB"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store PDF in vector DB")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    
@app.post("/send-to-qdrant/")
async def upload_to_qdrant(file: SendFilesRequest):
    """
    Endpoint to load the saved contents from PDF file, process the contents and send them to Qdrant.
    """
    filename = file.filename
    output_dir = "output"
    images_dir = f"output/{filename}_images"
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail=f"Documents file not found: {output_dir}")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"Images file not found: {images_dir}")

    try:
        with open(f"output/{filename}_content_list.json", "r", encoding="utf-8") as json_list:
            file_contents = json.load(json_list)
        
        pages = combine_page_contents(file_contents)

        chunks = chunk_pages(pages, 1500, 250) # Adjust as needed

        images = get_images(file_contents, f"{output_dir}/", app.state.embedding_models["clip"])
        # Send the loaded document chunks and images to Qdrant.
        success = send_to_qdrant(filename, chunks, images, app.state.embedding_models["txt"], file.collection)

        if success:
            return {"message": "File contents were uploaded to Qdrant successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file contents to Qdrant.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

# Endpoint to ask a question and retrieve the answer from the vector DB
@app.post("/ask-question/")
async def ask_question(question_request: QuestionRequest):
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """
    try:
       
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        txt_store, img_store = qdrant_client(app.state.embedding_models["txt"], app.state.embedding_models["clip"])
        
        # Get the question from the request body
        question = question_request.question

        # Use the question-answer retrieval function to get the response
        response, images = qa_ret(txt_store, img_store, question)
        print(f"{response}\n")
        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")

# A simple health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Success"}


# Endpoint to ask question and retrive answer to frontend streamlit
@app.post("/message")
async def ask_question_2(data: QuestionRequest = Body(None)):    
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """

    logger.debug(session)
    if 'item' not in session:
        session['item'] = ''
    
    #if not isinstance(data, dict):
    #    return "Could not handle request, data is not a dictionary"
    
    logger.debug(f"Data: {data}")
    
    try:  
        #response = agents[data['username']](data['input'])

        # Get the question from the request body
        question = data.question
        k = data.n_chunks
        collection = data.collection
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        txt_store, img_store = qdrant_client(app.state.embedding_models["txt"], app.state.embedding_models["clip"], collection)


        # Use the question-answer retrieval function to get the response
        response, images = qa_ret(txt_store, img_store, question, k)
        if images == "":
            data = {'ai_response': response, 'pages': ''}
        else:
            data = {'ai_response':response, 'image_base64':images[0].page_content, 'image_caption':images[0].metadata["image_caption"]}

        return data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")
    