import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ParsedQuery
from openai import OpenAI
import json

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set in .env")

# OpenRouter client
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",  # optional
        "X-Title": "QueryParserApp"
    }
)

class LLMParser:
    def __init__(self):
        self.client = client

    def parse(self, question: str) -> ParsedQuery:
        prompt = self._build_prompt(question)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query parser. You must respond ONLY in the following JSON format:\n"
                        "{\n"
                        '  "intent": string,\n'
                        '  "entities": [string],\n'
                        '  "conditions": {string: any},\n'
                        '  "original_query": string\n'
                        "}\n"
                        "No explanations or extra text. Just return valid JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()

        try:
            json_data = json.loads(content)
            return ParsedQuery(**json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Failed to parse LLM response as JSON:\n{content}\n\nError: {e}")

    def _build_prompt(self, question: str) -> str:
        return f"{question}"
