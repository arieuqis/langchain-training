from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# Path to the PDF file (same directory as this script)
pdf_path = "example_pdf.pdf"

# Load the PDF using PyPDFLoader
print(f"Loading content from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"First document preview: {docs[0].page_content[:200]}...")
print()

# Split the documents into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum characters per chunk
    chunk_overlap=200,  # Overlap between chunks to maintain context
    length_function=len,  # Function to measure chunk length
    separators=["\n\n", "\n", " ", ""]  # Separators to split on (in order of preference)
)

print("Splitting documents into chunks...")
chunks = text_splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")
print()

# Display the first few chunks
print("First 3 chunks:")
print("-" * 50)
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i + 1}:")
    print(f"Content: {chunk.page_content[:150]}...")
    print(f"Metadata: {chunk.metadata}")
    print()
