# utils/document_parser.py
import requests # Used for making HTTP requests to download files from URLs
import pypdf # Used for reading PDF files and extracting text
import os # Used for interacting with the operating system, specifically for file paths and deletion
from langchain.text_splitter import RecursiveCharacterTextSplitter # A sophisticated text splitter from Langchain
from typing import List, Dict # Type hints for better code readability and static analysis

def fetch_document(url: str) -> str:
    """Downloads a document from a URL and returns its local path."""
    local_filepath = "temp_policy.pdf" # Define a fixed temporary file name for the downloaded PDF
    print(f"Downloading document from {url}...") # Informative print statement
    try: # Start a try block to handle potential network or file-related errors during download
        response = requests.get(url, stream=True) # Make an HTTP GET request to the URL. 'stream=True' is crucial for large files; it allows processing the response content in chunks rather than loading it all into memory at once.
        response.raise_for_status()  # Check if the HTTP request was successful (status code 2xx). If not, it raises an HTTPError.
        with open(local_filepath, 'wb') as f: # Open the local file in write-binary mode ('wb') to save the PDF. 'with' ensures the file is properly closed even if errors occur.
            for chunk in response.iter_content(chunk_size=8192): # Iterate over the response content in chunks of 8192 bytes (8KB). This is efficient for memory.
                f.write(chunk) # Write each received chunk to the local file.
        print(f"✅ Document downloaded to {local_filepath}") # Success message
        return local_filepath # Return the path to the downloaded temporary file
    except requests.exceptions.RequestException as e: # Catch any exception specific to the 'requests' library (e.g., network errors, invalid URL, HTTP errors)
        print(f"❌ Error downloading document: {e}") # Print an error message
        raise # Re-raise the exception to propagate it up, indicating that the download failed and subsequent steps cannot proceed.

def extract_text_from_pdf(filepath: str) -> str:
    """Extracts text from a PDF file."""
    print(f"Extracting text from {filepath}...") # Informative print statement
    try: # Start a try block to handle potential errors during PDF parsing
        reader = pypdf.PdfReader(filepath) # Create a PdfReader object to parse the PDF file
        text = "" # Initialize an empty string to accumulate text from all pages
        for page in reader.pages: # Iterate through each page object in the PDF
            text += page.extract_text() or "" # Extract text from the current page. The 'or ""' ensures that if a page returns None (e.g., empty or image-only page), an empty string is used instead of raising an error when concatenating.
        print(f"✅ Extracted {len(text)} characters from PDF.") # Success message, showing the total character count
        return text # Return the concatenated text from all pages
    except Exception as e: # Catch any general exception that might occur during PDF processing (e.g., corrupted file, unreadable format)
        print(f"❌ Error extracting text from PDF: {e}") # Print an error message
        raise # Re-raise the exception to signal that text extraction failed.
    finally: # This 'finally' block will always execute, regardless of whether an exception occurred or not.
        # Clean up the downloaded file after we're done with it
        if os.path.exists(filepath): # Check if the temporary PDF file still exists
            os.remove(filepath) # Delete the temporary PDF file to clean up resources
            print(f"Cleaned up temporary file: {filepath}") # Confirmation message for cleanup

def chunk_text(text: str, doc_id: str = "doc1") -> List[Dict]:
    """Splits text into chunks and adds metadata."""
    print("Splitting text into chunks...") # Informative print statement
    text_splitter = RecursiveCharacterTextSplitter( # Initialize the text splitter
        chunk_size=500,  # The maximum size (in characters) of each text chunk. This is a configurable parameter; a good chunk size balances context and granularity.
        chunk_overlap=50, # The number of characters that will overlap between consecutive chunks. Overlap helps maintain context when a relevant piece of information spans across two chunk boundaries.
        length_function=len, # Specifies the function used to calculate the length of text (here, standard Python len() for character count).
    )
    # Langchain's splitter can create 'Documents', we just need the text content
    chunks_content = text_splitter.split_text(text) # Use the splitter to break the raw text into a list of strings (chunks).

    processed_chunks = [] # Initialize an empty list to store the final processed chunk dictionaries
    for i, chunk_text in enumerate(chunks_content): # Iterate through each chunk text along with its index (i)
        processed_chunks.append({ # Append a new dictionary for each chunk
            "id": f"{doc_id}-chunk-{i}",  # A unique ID for each chunk. It combines the document ID (e.g., "doc1") with the chunk's index (e.g., "doc1-chunk-0", "doc1-chunk-1"). This is critical for Pinecone and for managing chunks from different documents.
            "text": chunk_text, # The actual text content of the chunk
            "metadata": { # A nested dictionary for storing additional structured information about the chunk
                "doc_id": doc_id, # Stores the original document ID, allowing you to trace a chunk back to its source document
                "chunk_index": i # Stores the numerical index of the chunk within its original document, useful for ordering or reconstruction
            }
        })
    print(f"✅ Created {len(processed_chunks)} chunks.") # Success message, indicating the number of chunks generated
    return processed_chunks # Return the list of structured chunk dictionaries