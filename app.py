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

# Import the necessary functions from utils.py
from utils import process_pdf, send_to_qdrant, qdrant_client, qa_ret, get_callback_handler, get_embedding_models, process_pdf_with_tables


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

# Loading environment variables
load_dotenv(find_dotenv())


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", FRONTEND_URL],  # Allow requests from your React app (adjust domain if necessary)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)



class QuestionRequest(BaseModel):
    question: str


# Define a prompt for the API
class RAGChatPromptTemplate(ChatPromptTemplate):
    prompt:str
    
    def __init__(self, template:str):
        self.prompt = ChatPromptTemplate.from_template(template)


# Endpoint to upload a PDF and process it, sending to Qdrant
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file, process it, and store in the vector DB.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            filename = file.filename.split(".")[0]
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
            
        
        # Process the PDF to get document chunks and embeddings
        document_chunks, images = process_pdf_with_tables(temp_file_path, app.state.embedding_models["clip"])

        # Get the embedding model for text chunks
        #embedding_model = get_embedding_models()
        
        # Send the document chunks (with embeddings) to Qdrant
        success = send_to_qdrant(filename, document_chunks, images, app.state.embedding_models["txt"])
        
        # Remove the temporary file after processing
        os.remove(temp_file_path)

        if success:
            return {"message": "PDF successfully processed and stored in vector DB"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store PDF in vector DB")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

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
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        txt_store, img_store = qdrant_client(app.state.embedding_models["txt"], app.state.embedding_models["clip"])

        # Get the question from the request body
        #question = question_request.question
        question = data.question



        # Use the question-answer retrieval function to get the response
        response, images = qa_ret(txt_store, img_store, question)
        if images == "":
            data = {'ai_response': response, 'pages': ''}
        else:
            data = {'ai_response':response, 'image_base64':images[0].page_content, 'image_caption':images[0].metadata["image_caption"]}

        return data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")
    