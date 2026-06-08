from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from dotenv import load_dotenv
import os
load_dotenv()

# ============================================================================
# STEP 1: Configuration
# ============================================================================
# Connection string from environment variable
connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

# Get embedding model from environment variable, default to text-embedding-3-small
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Collection name (must match the one used during ingestion)
collection_name = "langchain_demo"

# ============================================================================
# STEP 2: Initialize Embeddings
# ============================================================================
print(f"Initializing OpenAI embeddings with model: {embedding_model}")
embeddings = OpenAIEmbeddings(model=embedding_model)

# ============================================================================
# STEP 3: Connect to Existing Vector Store
# ============================================================================
print(f"Connecting to vector store: {collection_name}")
vectorstore = PGVector(
    embedding_function=embeddings,
    connection_string=connection_string,
    collection_name=collection_name,
)

print("Connected successfully")
print()

# ============================================================================
# STEP 4: Semantic Search with Scores
# ============================================================================
# similarity_search_with_score() returns documents with their similarity scores
# Lower score = more similar (euclidean distance in vector space)
# This is useful for:
# - Filtering results by similarity threshold
# - Ranking results by relevance
# - Debugging search quality
print("Testing semantic search with scores...")
query = "preços planos hobby"  # Query in Portuguese
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

print(f"Found {len(results_with_scores)} results for query: '{query}'")
print("-" * 50)
for i, (doc, score) in enumerate(results_with_scores):
    print(f"Result {i + 1} (score: {score:.4f}):")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print()

# ============================================================================
# STEP 5: Search with Similarity Threshold
# ============================================================================
# Filter results by only keeping those below a certain score threshold
# This ensures only highly relevant results are returned
print("Testing search with similarity threshold...")
score_threshold = 0.5  # Only return results with score < 0.5
results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

# Filter by threshold
filtered_results = [(doc, score) for doc, score in results_with_scores if score < score_threshold]

print(f"Found {len(filtered_results)} results with score < {score_threshold}")
print("-" * 50)
for i, (doc, score) in enumerate(filtered_results):
    print(f"Result {i + 1} (score: {score:.4f}):")
    print(f"Content: {doc.page_content[:200]}...")
    print()

# ============================================================================
# STEP 6: Multiple Queries Example
# ============================================================================
print("Testing multiple queries...")
queries = [
    "preços planos hobby",
    "plano individual",
    "enterprise features"
]

for query in queries:
    results = vectorstore.similarity_search_with_score(query, k=2)
    print(f"Query: '{query}'")
    print(f"Top result score: {results[0][1]:.4f}")
    print(f"Top result preview: {results[0][0].page_content[:100]}...")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 50)
print("Search capabilities demonstrated:")
print("1. Basic similarity search with scores")
print("2. Filtering results by similarity threshold")
print("3. Multiple query search")
print("=" * 50)
