import os
import json
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from models.llm import embeddings
import faiss
from typing import Dict
from langchain.docstore.in_memory import InMemoryDocstore

# Define paths from environment variables
JSON_DIR = Path(os.getenv("JSON_DIR", "data/source"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "data/processed"))
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store/faiss_index"))

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

# Helper function to flatten nested dictionaries
def flatten_dict(data):
    """Recursively flattens a dictionary."""
    flat_items = []
    for key, value in data.items():
        if isinstance(value, dict):
            flat_items.extend(flatten_dict(value).items())
        else:
            flat_items.append((key, value))
    return dict(flat_items)

# Process JSON files
def process_json_files():
    for json_file in JSON_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_entries = []

        # Handle different data structures
        if isinstance(data, list):
            for entry in data:
                flattened_entry = flatten_dict(entry)
                content = "\n".join([f"{k}: {v}" for k, v in flattened_entry.items() if v])
                split_docs = splitter.split_text(content)
                processed_entries.extend([{"page_content": doc, "metadata": {}} for doc in split_docs])

        elif isinstance(data, dict):
            flattened_data = flatten_dict(data)
            content = "\n".join([f"{k}: {v}" for k, v in flattened_data.items() if v])
            split_docs = splitter.split_text(content)
            processed_entries.extend([{"page_content": doc, "metadata": {}} for doc in split_docs])

        else:
            print(f"Unsupported structure in file: {json_file}")
            continue

        # Save processed documents
        output_file = PROCESSED_DIR / f"{json_file.stem}_processed.json"
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(processed_entries, out_f, ensure_ascii=False, indent=4)
            print(f"Processed and saved {output_file}")

# Load processed files
def load_processed_files():
    docs = []

    # Check if there are files in the processed directory
    if not any(PROCESSED_DIR.glob("*.json")):
        print("No processed files found. Running process_json_files...")
        process_json_files()

    # Load processed files
    for json_file in PROCESSED_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs.extend([Document(page_content=entry["page_content"], metadata=entry.get("metadata", {})) for entry in data])

    print(f"Loaded {len(docs)} documents.")
    return docs

# Create a new FAISS vector store
def create_vector_store(embeddings):
    """
    Create a FAISS vector store with the specified embeddings.
    """
    # Dynamically determine the embedding dimension
    sample_embedding = embeddings.embed_query("Sample text to determine dimension")
    dimension = len(sample_embedding)

    # Initialize a FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Create an in-memory docstore
    docstore = InMemoryDocstore({})

    # Create an empty mapping of index to docstore ID
    index_to_docstore_id = {}

    # Initialize the FAISS vector store
    return FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embeddings.embed_query  # Pass the embedding function here
    )

# Embed documents and save the vector store
def embed_and_store_documents():
    if VECTOR_STORE_PATH.exists() and (VECTOR_STORE_PATH / "index.faiss").is_file():
        print(f"Vector store already exists at {VECTOR_STORE_PATH}. Skipping embedding.")
        return

    # Process and load documents
    process_json_files()
    docs = load_processed_files()

    # Create FAISS vector store
    vector_store = create_vector_store(embeddings)

    # Index documents
    vector_store.add_documents(docs)

    # Save the FAISS vector store to disk
    vector_store.save_local(str(VECTOR_STORE_PATH))
    print(f"Vector store saved at {VECTOR_STORE_PATH}.")

# Load an existing FAISS vector store
def load_vector_store():
    return FAISS.load_local(str(VECTOR_STORE_PATH), embeddings)

# Initialize the vector store
def initialize_vector_store():
    """
    Initialize the FAISS vector store by loading or creating it.
    """
    if VECTOR_STORE_PATH.exists() and (VECTOR_STORE_PATH / "index.faiss").is_file():
        print("Loading existing vector store...")
        return FAISS.load_local(
            str(VECTOR_STORE_PATH), 
            #embeddings.embed_query, 
            embeddings,
            allow_dangerous_deserialization=True  # Enable safe deserialization
        )
    else:
        print("Vector store not found. Creating a new one...")
        embed_and_store_documents()
        return FAISS.load_local(
            str(VECTOR_STORE_PATH), 
            embeddings.embed_query, 
            allow_dangerous_deserialization=True  # Enable safe deserialization
        )

initialize_vector_store()
