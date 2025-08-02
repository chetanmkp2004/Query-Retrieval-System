# services/answer_generator.py
from typing import List, Dict
import os
import openai
from dotenv import load_dotenv

class AnswerGenerator:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENROUTER_API_KEY")
        openai.api_base = "https://openrouter.ai/api/v1"  # or OpenAI's endpoint
        print("--- AnswerGenerator initialized ---")

    def generate_answer(self, question: str, parsed_query: Dict, retrieved_chunks: List[Dict]) -> str:
        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
        if not context.strip():
            return "No context provided. Please check document retrieval step."

        system_prompt = (
            "You are an expert policy analyst.\n"
            "Answer the user's question ONLY using the provided context below.\n"
            "If the answer is not in the context, say 'The information is not available in the provided document.'\n"
            "Cite the exact sentences from the context that support your answer."
        )

        user_prompt = (
            "=== CONTEXT ===\n"
            f"{context}\n\n"
            "=== QUESTION ===\n"
            f"{question}\n\n"
            "=== INSTRUCTIONS ===\n"
            "Use ONLY the above context to answer. Cite sentences if applicable."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # or replace with a working model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            answer = response['choices'][0]['message']['content'].strip()
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
