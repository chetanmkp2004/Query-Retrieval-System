from pydantic import BaseModel
from typing import List, Dict, Any

class QuestionRequest(BaseModel):
    documents: str  # URL of the document
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# Your model for holding the parsed information from a question
class ParsedQuery(BaseModel):
    intent: str  # e.g., "coverage_check", "define_term", "get_period"
    entities: List[str]  # e.g., ["knee surgery", "maternity expenses"]
    # A place for extra details, like a specific time period
    conditions: Dict[str, Any] = {}
    # Always keep the original question for context
    original_query: str
