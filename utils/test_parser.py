# test_parser.py
from document_parser import fetch_document, extract_text_from_pdf, chunk_text # Import the functions you just implemented

# Convert Google Drive view URL to direct download URL
# Original: https://drive.google.com/file/d/1g9W8hiKWYn7oIKmXxB2UiQWjUNI_BDOV/view?usp=drive_link
# Direct download format: https://drive.google.com/uc?export=download&id=FILE_ID
SAMPLE_URL = "https://drive.google.com/uc?export=download&id=1g9W8hiKWYn7oIKmXxB2UiQWjUNI_BDOV" # The specific PDF URL to use for testing

print("--- Testing Document Parser ---") # Header for the test output

try:
    filepath = fetch_document(SAMPLE_URL) # Call fetch_document to download the PDF and get its local path
    print(f"✅ Document downloaded successfully to: {filepath}")
    
    text = extract_text_from_pdf(filepath) # Call extract_text_from_pdf to get the full text from the downloaded PDF
    print(f"✅ Text extracted successfully. Length: {len(text)} characters")
    
    chunks = chunk_text(text) # Call chunk_text to split the extracted text into a list of chunk dictionaries
    print(f"✅ Text chunked successfully. Total chunks: {len(chunks)}")

    print(f"\n--- Sample of first chunk ---") # Header for the sample chunk output
    print(chunks[0]) # Print the first chunk dictionary to verify its structure (id, text, metadata) and content
    
    # Additional info
    print(f"\n--- Summary ---")
    print(f"Total chunks created: {len(chunks)}")
    if len(chunks) > 0:
        print(f"First chunk text preview: {chunks[0]['text'][:200]}...")
        if len(chunks) > 1:
            print(f"Second chunk text preview: {chunks[1]['text'][:200]}...")

except Exception as e:
    print(f"❌ Test failed: {e}")
    print("\nMake sure:")
    print("1. The Google Drive file is set to 'Anyone with the link can view'")
    print("2. The file ID in the URL is correct")
    print("3. The document_parser.py file has all required functions implemented")