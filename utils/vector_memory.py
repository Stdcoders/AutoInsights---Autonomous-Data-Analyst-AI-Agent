import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Persistent Chroma DB directory
CHROMA_DIR = os.path.join("memory", "chroma")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # lightweight & fast
    # You can swap with "all-mpnet-base-v2" for higher accuracy
)

# Create / load Chroma persistent store
vectorstore = Chroma(
    collection_name="ai_data_analyst_memory",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

def store_memory(text: str, metadata: dict = None):
    """Store a conversation or insight in vector DB."""
    vectorstore.add_texts([text], metadatas=[metadata or {}])
    vectorstore.persist()

def search_memory(query: str, top_k: int = 5):
    """Retrieve most relevant past memories for context."""
    results = vectorstore.similarity_search(query, k=top_k)
    return results
