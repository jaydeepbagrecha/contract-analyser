"""
Document Ingestion Pipeline
Loads documents, splits them into chunks, embeds them, and stores
in ChromaDB. This is the offline indexing stage of the RAG pipeline.
 
Pattern: Load -> Split -> Embed -> Store
"""
 
import os
from pathlib import Path
from dotenv import load_dotenv
 
# LangChain document loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
 
# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
# Embeddings and vector store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
 
load_dotenv()
 
# --- Configuration ---
#CHROMA_PERSIST_DIR = "chroma_db"
# Use /tmp on Streamlit Cloud (read-only filesystem), local path otherwise
if os.path.exists("/tmp/chroma_db"):
    CHROMA_PERSIST_DIR = "/tmp/chroma_db"
else:
    CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "contracts"
CHUNK_SIZE = 1000       # Characters per chunk
CHUNK_OVERLAP = 200     # Overlap between consecutive chunks
EMBEDDING_MODEL = "text-embedding-3-small"
 
 
# --- Step 1: Load Documents ---
def load_document(file_path: str) -> list:
    """
    Load a single document and return a list of LangChain Document objects.
    Each object has .page_content (text) and .metadata (source, page, etc.).
    """
    file_path = str(file_path)
    extension = file_path.lower().split(".")[-1]
 
    if extension == "pdf":
        loader = PyPDFLoader(file_path)
    elif extension in ("docx", "doc"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
 
    documents = loader.load()
 
    # Enrich metadata
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = filename
        doc.metadata["file_path"] = file_path
 
    print(f"  Loaded {filename}: {len(documents)} pages")
    return documents
 
 
def load_directory(directory: str) -> list:
    """Load all supported documents from a directory."""
    all_docs = []
    supported = (".pdf", ".docx", ".doc")
    dir_path = Path(directory)
 
    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in supported:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
 
    print(f"\nTotal: {len(all_docs)} pages from {directory}")
    return all_docs
 
 
# --- Step 2: Split into Chunks ---
def split_documents(documents: list) -> list:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    This splitter tries to split on natural boundaries:
    paragraphs (\n\n) first, then sentences (. ), then words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
 
    chunks = splitter.split_documents(documents)
 
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
 
    print(f"Split into {len(chunks)} chunks")
    print(f"  Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    return chunks
 
 
# --- Step 3 & 4: Embed and Store ---
def create_vector_store(chunks: list) -> Chroma:
    """Generate embeddings for all chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
 
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
 
    print(f"Stored {len(chunks)} embeddings in ChromaDB")
    print(f"  Persist directory: {CHROMA_PERSIST_DIR}")
    return vector_store
 
 
def load_vector_store() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
 
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
 
    count = vector_store._collection.count()
    print(f"Loaded vector store with {count} embeddings")
    return vector_store
 
 
# --- Full Pipeline ---
def ingest_documents(directory: str) -> Chroma:
    """Run the complete ingestion pipeline: Load -> Split -> Embed -> Store"""
    print("=" * 50)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 50)
 
    print("\n[1/3] Loading documents...")
    documents = load_directory(directory)
    if not documents:
        print("No documents found. Check the directory path.")
        return None
 
    print("\n[2/3] Splitting into chunks...")
    chunks = split_documents(documents)
 
    print("\n[3/3] Embedding and storing...")
    vector_store = create_vector_store(chunks)
 
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    return vector_store
 
 
if __name__ == "__main__":
    vs = ingest_documents("data/sample_contracts")
    if vs:
        results = vs.similarity_search("termination clause", k=3)
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Text: {doc.page_content[:200]}...")
