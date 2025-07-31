import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_parser import LLMParser
load_dotenv()

def test_questions():
    parser = LLMParser()

    questions = [
        "What was the revenue of Google in Q1 2023?",
        "List the top 3 products by sales in 2022.",
        "Show me the growth rate of Apple over the past 5 years."
    ]

    print("📤 Sending test questions to LLMParser...\n")

    for question in questions:
        try:
            parsed = parser.parse(question)
            print(f"🔹 Question: {question}")
            print(f"✅ Parsed Output: {parsed}\n")
        except Exception as e:
            print(f"❌ Failed to parse: '{question}'")
            print(f"   Error: {e}\n")

if __name__ == "__main__":
    test_questions()
