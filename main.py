# main.py
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from utils.document_parser import fetch_document, extract_text_from_pdf, chunk_text
from services.vector_store import VectorStoreManager

# Initialize the FastAPI app
app = FastAPI(title="HackRx LLM Retrieval System")

# Initialize our services (they will be shared across all requests)
vector_store_manager = VectorStoreManager()

# --- Models: Defines the structure of our JSON requests and responses ---
class QuestionRequest(BaseModel):
    documents: str  # URL of the document
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# --- Security ---
# This is a simple security check. The request must have a specific "Bearer Token".
security = HTTPBearer()
# This is the secret token required to access the API. Keep it safe.
# In a real app, this would be in a .env file.
AUTH_TOKEN = "e33ecffb686ac4409ef84a4cebb13b83e58b120976b3e7b73481e7cc3daf20de"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the token from the request header."""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )
    return credentials

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"message": "LLM-Powered Query-Retrieval System is running!"}

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def run_submission(request: QuestionRequest, token: str = Depends(verify_token)):
    """
    Main endpoint to process documents and answer questions.
    """
    print(f"Received request for document: {request.documents}")

    # --- Step 1: Document Processing (Integration with Member 3's work) ---
    # These functions are just skeletons for now. Member 3 will build the real ones.
    temp_filepath = fetch_document(request.documents)
    document_text = extract_text_from_pdf(temp_filepath)
    chunks = chunk_text(document_text)
    print(f"Processed {len(chunks)} chunks from the document.")

    # --- Step 2: Indexing (Integration with Member 3's work) ---
    # Once chunks are ready, we index them in our vector store.
    vector_store_manager.index_document_chunks(chunks)
    print("Document indexing process initiated.")

    # This is a placeholder. We will build the real logic here in the coming days.
    dummy_answers = [f"Dummy answer for question {i+1}" for i in range(len(request.questions))]

    return AnswerResponse(answers=dummy_answers)
