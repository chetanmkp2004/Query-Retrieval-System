import os
from dotenv import load_dotenv
import openai

load_dotenv()
api_base = "https://openrouter.ai/api/v1"
api_key = os.getenv("OPENROUTER_API_KEY")

client = openai.OpenAI(
  base_url=api_base,
  api_key=api_key,
  default_headers={ 
    "HTTP-Referer": "YOUR_SITE_URL", 
    "X-Title": "YOUR_APP_NAME",
  },
)


print("Sending a test request to OpenRouter...")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    answer = response.choices[0].message.content
    print(f"\nOpenRouter Response: {answer}")
    print("\n✅ API connection is successful!")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")