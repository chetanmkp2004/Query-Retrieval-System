from pydantic import BaseModel
from typing import List, Dict, Any

# (The Team Lead will add QuestionRequest and AnswerResponse here)

# Your model for holding the parsed information from a question
class ParsedQuery(BaseModel):
    intent: str  # e.g., "coverage_check", "define_term", "get_period"
    entities: List[str]  # e.g., ["knee surgery", "maternity expenses"]
    # A place for extra details, like a specific time period
    conditions: Dict[str, Any] = {}
    # Always keep the original question for context
    original_query: str
