from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()

# ============================================================================
# STEP 1: Configuration
# ============================================================================
# Path to the PDF file
pdf_path = "5-loaders-and-vector-databases/example_pdf.pdf"

# Connection string from environment variable
# Format: postgresql://user:password@host:port/database
connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

# ============================================================================
# STEP 2: Document Loading
# ============================================================================
# PyPDFLoader reads the PDF and returns a list of documents
# Each page of the PDF becomes a separate document
print(f"Loading content from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print()

# ============================================================================
# STEP 3: Splitting into Chunks
# ============================================================================
# RecursiveCharacterTextSplitter splits documents into smaller pieces
# This is necessary because:
# - LLMs have context limits (tokens)
# - Smaller chunks allow more precise search
# - Overlap maintains context between adjacent chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum size of each chunk (characters)
    chunk_overlap=200,      # Overlap between chunks to maintain context
    length_function=len,    # Function to measure chunk size
    separators=["\n\n", "\n", " ", ""]  # Separators to split (in order of preference)
)

print("Splitting documents into chunks...")
chunks = text_splitter.split_documents(docs)

# Enrich chunks by filtering metadata to remove empty values
# This ensures only meaningful metadata is stored in the vector database
enriched_chunks = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in chunks
]

# Create custom IDs for each chunk (doc_0, doc_1, doc_2, ...)
# This allows tracking and managing individual documents
chunk_ids = [f"doc_{i}" for i in range(len(enriched_chunks))]

print(f"Created {len(enriched_chunks)} enriched chunks with custom IDs")
print()

# ============================================================================
# STEP 4: Embedding Generation
# ============================================================================
# Embeddings are numerical representations of text in vector space
# Texts with similar meanings are close together in vector space
# Get embedding model from environment variable, default to text-embedding-3-small
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
print(f"Initializing OpenAI embeddings with model: {embedding_model}")
embeddings = OpenAIEmbeddings(model=embedding_model)

# ============================================================================
# STEP 5: Vector Store Creation (Indexing)
# ============================================================================
# PGVector stores embeddings in PostgreSQL with pgvector extension
# Each chunk is converted to embedding and stored with its metadata
# This enables efficient semantic search by vector similarity
print("Creating PGVector store...")
collection_name = "langchain_demo"  # Name of the collection/table in the database

# Create PGVector instance (constructor approach)
# This allows more control and flexibility for adding documents
vectorstore = PGVector(
    embedding_function=embeddings,
    connection_string=connection_string,
    collection_name=collection_name,
)

# Add documents with custom IDs
# This allows tracking each document individually
print("Adding documents to vector store...")
vectorstore.add_documents(enriched_chunks, ids=chunk_ids)

print(f"Vector store created with collection: {collection_name}")
print(f"Successfully indexed {len(enriched_chunks)} documents")
print()

# ============================================================================
# RAG (Retrieval-Augmented Generation) PIPELINE SUMMARY
# ============================================================================
# 1. Loading: Load documents (PDF, web, etc.)
# 2. Splitting: Split into smaller chunks
# 3. Embedding: Convert chunks to numerical vectors
# 4. Indexing: Store embeddings in vector database
# 5. Retrieval: (Next file) Search for chunks similar to query
# 6. Generation: (Next step) Send chunks + query to LLM to generate response
# ============================================================================
