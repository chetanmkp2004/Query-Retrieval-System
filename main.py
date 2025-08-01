# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import the models from your new models.py file
from models import QuestionRequest, AnswerResponse
from utils.document_parser import fetch_document, extract_text_from_pdf, chunk_text
from services.vector_store import VectorStoreManager
from services.llm_parser import LLMParser
from services.answer_generator import AnswerGenerator

app = FastAPI(title="HackRx LLM Retrieval System")

# Initialize our services (they will be shared across all requests)
vector_store_manager = VectorStoreManager()
llm_parser = LLMParser()
answer_generator = AnswerGenerator()

# --- Security ---
security = HTTPBearer()
AUTH_TOKEN = "e33ecffb686ac4409ef84a4cebb13b83e58b120976b3e7b73481e7cc3daf20de"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )
    return credentials

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "LLM-Powered Query-Retrieval System is running!"}

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def run_submission(request: QuestionRequest, token: str = Depends(verify_token)):
    """
    Main endpoint to process documents and answer questions.
    """
    print(f"Received request for document: {request.documents}")

    # --- Step 1: Document Processing (Integration with Member 3's work) ---
    temp_filepath = fetch_document(request.documents)
    document_text = extract_text_from_pdf(temp_filepath)
    chunks = chunk_text(document_text)
    print(f"Processed {len(chunks)} chunks from the document.")

    # --- Step 2: Indexing (Integration with Member 3's work) ---
    vector_store_manager.index_document_chunks(chunks)
    print("Document indexing process initiated.")

    # --- Step 3: Query Processing and Answering ---
    final_answers = []
    for question in request.questions:
        # 1. Parse the query (Member 2's work)
        parsed_query = llm_parser.parse_query(question)

        # 2. Retrieve relevant chunks from vector DB (Member 3's work)
        retrieved_chunks = vector_store_manager.query_vectors(question)

        # 3. Generate the answer using the chunks (Member 2's work)
        final_answer_text = answer_generator.generate_answer(question, parsed_query, retrieved_chunks)

        final_answers.append(final_answer_text)

    return AnswerResponse(answers=final_answers)
