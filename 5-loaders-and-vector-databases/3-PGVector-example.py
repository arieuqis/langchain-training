from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from dotenv import load_dotenv
import os
load_dotenv()

# Path to the PDF file
pdf_path = "5-loaders-and-vector-databases/example_pdf.pdf"

# Connection string from environment variable
connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

# Load the PDF using PyPDFLoader
print(f"Loading content from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print()

# Split the documents into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

print("Splitting documents into chunks...")
chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")
print()

# Initialize OpenAI embeddings
print("Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings()

# Create PGVector store with the chunks
print("Creating PGVector store...")
collection_name = "langchain_demo"

# Create the vector store (this will create the table if it doesn't exist)
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string=connection_string,
    collection_name=collection_name,
)

print(f"Vector store created with collection: {collection_name}")
print()

# Test similarity search
print("Testing similarity search...")
query = "preços planos hobby"
results = vectorstore.similarity_search(query, k=2)

print(f"Found {len(results)} results for query: '{query}'")
print("-" * 50)
for i, doc in enumerate(results):
    print(f"Result {i + 1}:")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print()

# Test similarity search with scores
print("Testing similarity search with scores...")
results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

print(f"Found {len(results_with_scores)} results with scores")
print("-" * 50)
for i, (doc, score) in enumerate(results_with_scores):
    print(f"Result {i + 1} (score: {score:.4f}):")
    print(f"Content: {doc.page_content[:200]}...")
    print()
