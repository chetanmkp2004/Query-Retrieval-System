import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "hackrx-embed" # <-- UPDATED: Your specific Pinecone index name

if not API_KEY or not ENVIRONMENT:
    print("❌ Pinecone credentials not found in .env file.")
else:
    try:
        pc = Pinecone(api_key=API_KEY)
        
        # Check if the specific index exists in your Pinecone project
        if INDEX_NAME in [index.name for index in pc.list_indexes()]:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            print("✅ Successfully connected to Pinecone!")
            print(f"Index stats: {stats}")
            
            # --- IMPORTANT: Verify Dimensions Here ---
            if stats.get('dimension') != 1024: # <-- IMPORTANT: Check for 1024 dimensions now
                print(f"⚠️ WARNING: Index '{INDEX_NAME}' has dimensions {stats.get('dimension')}, expected 1024 for llama-text-embed-v2 model.")
                print("Please delete and recreate the Pinecone index with 1024 dimensions.")
            else:
                print("Dimensions match expected model (1024).")
        else:
            print(f"❌ Index '{INDEX_NAME}' does not exist. Please create it in the Pinecone dashboard.")
    except Exception as e:
        print(f"❌ An error occurred connecting to Pinecone: {e}")