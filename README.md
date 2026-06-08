# LangChain Training

This repository contains practical examples for learning LangChain, organized by topic. This guide is designed for developers coming from Java/Kotlin (Spring Boot), Node.js/TypeScript, or Flutter backgrounds.

## Python Quick Start for Java/Kotlin/Node.js Developers

### Key Python Concepts (for your background)

**Python vs Java/Kotlin:**
- **Interpreted language**: No compilation step (like Node.js), runs directly
- **Dynamic typing**: Variable types inferred at runtime (like JavaScript/TypeScript without strict mode)
- **Indentation matters**: No braces `{}`, uses indentation for code blocks (like Python's version of Kotlin's significant whitespace)
- **dict**: Key-value structure (equivalent to JSON objects in JavaScript, Map in Java/Kotlin)
- **No semicolons**: Line endings define statements

**Python Ecosystem:**
- **pip**: Python package manager (like npm for Node.js, Maven/Gradle for Java)
- **venv**: Virtual environment (like Node.js's node_modules isolation, Java's module systems)
- **requirements.txt**: Dependency list (like package.json in Node.js, pom.xml/build.gradle in Java)
- **.py files**: Python source files (like .js in Node.js, .kt in Kotlin, .java in Java)

### Key Libraries Used

**LangChain Libraries:**
- **langchain-core**: Core LangChain abstractions and interfaces
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-google-genai**: Google AI integration (Gemini)

**Python Utilities:**
- **python-dotenv**: Loads environment variables from .env files (like dotenv in Node.js)
- **pydantic**: Data validation using Python type annotations (like Zod in TypeScript, validation in Java)

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (or Google API key for some examples)

## Setup

### 1. Create Virtual Environment

**What is venv?**
Virtual environment creates an isolated Python environment with its own installed packages. This is similar to:
- **Node.js**: Each project having its own node_modules
- **Java**: Using different JDK versions or dependency scopes
- **Kotlin**: Gradle's dependency management per project
- **Docker**: Like a lightweight container that isolates dependencies (but without the full OS isolation)

```bash
python -m venv venv
```

**Explanation:**
- `python`: Python interpreter
- `-m venv`: Run the venv module as a script
- `venv`: Name of the virtual environment directory (can be any name)

This creates a `venv/` directory with:
- Python interpreter copy
- pip package manager
- Isolated package installation directory

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

**What this does:**
- Modifies your PATH to use the virtual environment's Python
- Subsequent `python` and `pip` commands use the isolated environment
- Your terminal prompt usually changes to show `(venv)` prefix

**Similar to:**
- **Node.js**: Not directly equivalent, but similar to using nvm to switch Node versions
- **Java**: Using different JAVA_HOME for different projects

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**What is pip?**
pip is Python's package installer, similar to:
- **npm**: Node.js package manager
- **Maven/Gradle**: Java dependency management tools
- **pub**: Flutter/Dart package manager

**What is requirements.txt?**
Text file listing all dependencies with versions, similar to:
- **package.json**: Node.js dependencies
- **pom.xml**: Maven dependencies
- **build.gradle**: Gradle dependencies

**Example requirements.txt:**
```
langchain-core==0.1.0
langchain-openai==0.0.5
python-dotenv==1.0.0
```

The `==` pins exact versions (like `=1.0.0` in package.json or strict version ranges in Maven).

### 4. Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

**What is .env?**
File for environment variables, similar to:
- **Node.js**: .env files used with dotenv
- **Java/Kotlin**: application.properties or application.yml
- **Flutter**: .env files (with flutter_dotenv)

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your_actual_openai_key_here
GOOGLE_API_KEY=your_actual_google_api_key_here
```

**What is python-dotenv?**
Library that loads variables from .env into `os.environ` (Python's environment variable dictionary), similar to:
- **Node.js**: `dotenv.config()`
- **Java**: `System.getenv()` or Spring's `@Value`
- **Kotlin**: Similar to Java, or using environment config libraries

## Running the Examples

### Activate the virtual environment first:
```bash
source venv/bin/activate
```

### 1-basics - LangChain Fundamentals

```bash
python 1-basics/1-hello-world.py
python 1-basics/2-init-chat-model.py
python 1-basics/3-prompt-templates.py
python 1-basics/4-chat-prompts.py
```

**What each file teaches:**

**1-hello-world.py**: Basic LangChain setup and simple invocation
- Similar to "Hello World" in any language
- Shows how to initialize and use a LangChain component

**2-init-chat-model.py**: Chat model initialization
- Shows how to create a chat model instance
- Similar to initializing a database connection or API client in Java/Spring

**3-prompt-templates.py**: Prompt templates
- Demonstrates parameterized prompts (like string templates in any language)
- Similar to prepared statements in SQL, or template engines in web frameworks

**4-chat-prompts.py**: Chat-specific prompts
- Shows chat message types (system, user, assistant)
- Similar to message objects in chat applications

### 2-chains-and-processing - Chains and Processing

```bash
python 2-chains-and-processing/1-starting-using-chains.py
python 2-chains-and-processing/2-chains-with-decorators.py
python 2-chains-and-processing/runnable-lambda.py
python 2-chains-and-processing/4-pipeline-example.py
```

**What each file teaches:**

**1-starting-using-chains.py**: Basic chain usage with pipe operator
- Introduces LCEL (LangChain Expression Language)
- The `|` operator chains components (output of left becomes input of right)
- Similar to:
  - **Java Streams**: `.map().filter().collect()`
  - **RxJava/Reactor**: Operator chains
  - **Node.js streams**: `.pipe()`
  - **Kotlin Flow**: Operator chains

**2-chains-with-decorators.py**: Using @chain decorator
- Python decorators modify functions (like annotations in Java/Kotlin)
- The `@chain` decorator transforms functions into LangChain Runnables
- Similar to:
  - **Java**: `@Component`, `@Service` annotations
  - **Kotlin**: Similar annotation system
  - **TypeScript**: Decorators (experimental)
  - **Flutter**: Not directly equivalent, but similar to widget composition

**runnable-lambda.py**: RunnableLambda wrapper
- Alternative to @chain decorator
- Wraps functions explicitly (more verbose but clearer)
- Similar to:
  - **Java**: Wrapping functions in Runnable/Supplier interfaces
  - **Kotlin**: Lambda expressions
  - **Node.js**: Wrapping functions for middleware
  - **TypeScript**: Higher-order functions

**4-pipeline-example.py**: Multi-step pipeline
- Demonstrates complex chain composition
- Shows output parsers (StrOutputParser)
- Similar to:
  - **Java Spring**: Filter chains, interceptor patterns
  - **Node.js Express**: Middleware chains
  - **Kotlin**: Coroutines/Flow pipelines
  - **Flutter**: Stream transformations

### 4-memory-management - Memory Management

```bash
python 4-memory-management/1-simple-history-example.py
python 4-memory-management/2-sliding-window-history-example.py
```

**What each file teaches:**

**1-simple-history-example.py**: Basic chat history management
- Demonstrates `InMemoryChatMessageHistory` for storing conversation history
- Shows `RunnableWithMessageHistory` for automatic session-based memory
- Similar to:
  - **Java Spring**: Session management, conversation scope
  - **Node.js**: Session middleware, Redis-based sessions
  - **Kotlin**: State management in coroutines

**2-sliding-window-history-example.py**: Sliding window memory with trimming
- Uses `trim_messages` to keep only recent messages in history
- Combines with `RunnableLambda` for input preparation
- Integrates with `RunnableWithMessageHistory` for session management
- Similar to:
  - **Java**: Circular buffers, LRU caches
  - **Node.js**: Rate limiting with sliding windows
  - **Kotlin**: Flow with buffer operators

### 5-loaders-and-vector-databases - Document Loaders and Vector Databases

**Prerequisites:**
```bash
# Start PostgreSQL with pgvector extension
docker-compose up -d
```

```bash
python 5-loaders-and-vector-databases/1-WebBaseLoader-example.py
python 5-loaders-and-vector-databases/2-PyPDFLoader-example.py
python 5-loaders-and-vector-databases/3-PGVector-ingestion.py
python 5-loaders-and-vector-databases/4-PGVector-search-example.py
```

**What each file teaches:**

**1-WebBaseLoader-example.py**: Web document loading
- Demonstrates loading content from websites using `WebBaseLoader`
- Splits documents into chunks using `RecursiveCharacterTextSplitter`
- Similar to:
  - **Java**: Jsoup for web scraping
  - **Node.js**: Cheerio, Puppeteer
  - **Kotlin**: Ktor web client

**2-PyPDFLoader-example.py**: PDF document loading
- Shows how to load and parse PDF files using `PyPDFLoader`
- Splits PDF content into searchable chunks
- Similar to:
  - **Java**: Apache PDFBox, iText
  - **Node.js**: pdf-parse, pdf-lib
  - **Kotlin**: Apache PDFBox wrapper

**3-PGVector-ingestion.py**: Vector database ingestion (RAG - Part 1)
- Demonstrates the RAG ingestion pipeline:
  1. Loading documents (PDF)
  2. Splitting into chunks
  3. Generating embeddings with OpenAI
  4. Enriching metadata (filtering empty values)
  5. Storing in PostgreSQL with pgvector extension
- Uses custom document IDs for tracking
- Similar to:
  - **Java Spring**: Elasticsearch indexing, document repositories
  - **Node.js**: MongoDB indexing, vector databases
  - **Kotlin**: Exposed ORM, database migrations

**4-PGVector-search-example.py**: Vector database search (RAG - Part 2)
- Demonstrates semantic search with `similarity_search_with_score`
- Shows filtering by similarity threshold
- Multiple query examples
- Similar to:
  - **Java Spring**: Elasticsearch queries, similarity search
  - **Node.js**: MongoDB aggregation, vector search
  - **Kotlin**: Query DSL, database filtering

**Key Concepts:**
- **Embeddings**: Numerical representations of text for semantic search
- **Vector Database**: PostgreSQL with pgvector extension for storing embeddings
- **RAG (Retrieval-Augmented Generation)**: Combining document retrieval with LLM generation
- **Similarity Search**: Finding documents by meaning, not just keywords

## Python Syntax Notes for Java/Kotlin/Node.js Developers

### Type Hints
```python
def square(x: int) -> int:
    return x * x
```
- `: int` - parameter type hint (like TypeScript, Kotlin)
- `-> int` - return type hint (like TypeScript, Kotlin)
- **Optional at runtime** - Python doesn't enforce these (unlike Java/Kotlin)
- Similar to TypeScript's type system (optional but helpful)

### Decorators
```python
@chain
def my_function():
    pass
```
- Decorators modify functions (like Java annotations, but executable)
- Similar to:
  - **TypeScript**: `@decorator` (experimental)
  - **Java**: `@Annotation` (but decorators are more powerful)
  - **Kotlin**: Similar annotation system

### Dictionary (dict)
```python
my_dict = {"key": "value", "x": 5}
value = my_dict["x"]
```
- Key-value structure (like JSON objects, Java Map, Kotlin Map)
- Access with square brackets (like array/object access in JavaScript)
- Similar to:
  - **JavaScript**: `const obj = {key: "value"}`
  - **Java**: `Map<String, Object> map = new HashMap<>()`
  - **Kotlin**: `val map = mapOf("key" to "value")`

### List Comprehensions (if you see them)
```python
squares = [x * x for x in range(10)]
```
- Concise way to create lists (like map/filter in functional programming)
- Similar to:
  - **JavaScript**: `[...Array(10).keys()].map(x => x * x)`
  - **Java**: streams `.map().collect()`
  - **Kotlin**: `(1..10).map { it * it }`

## Project Structure

```
langchain-training/
├── 1-basics/                    # Basic LangChain concepts
│   ├── 1-hello-world.py        # Simple LangChain example
│   ├── 2-init-chat-model.py    # Chat model initialization
│   ├── 3-prompt-templates.py   # Prompt templates
│   └── 4-chat-prompts.py        # Chat-specific prompts
├── 2-chains-and-processing/     # Chain composition and processing
│   ├── 1-starting-using-chains.py   # Basic chain usage
│   ├── 2-chains-with-decorators.py   # Using @chain decorator
│   ├── runnable-lambda.py            # RunnableLambda wrapper
│   └── 4-pipeline-example.py         # Multi-step pipeline
├── 4-memory-management/         # Chat history and memory management
│   ├── 1-simple-history-example.py       # Basic history management
│   └── 2-sliding-window-history-example.py  # Sliding window memory
├── 5-loaders-and-vector-databases/  # Document loaders and vector databases
│   ├── 1-WebBaseLoader-example.py       # Web document loading
│   ├── 2-PyPDFLoader-example.py         # PDF document loading
│   ├── 3-PGVector-ingestion.py          # Vector database ingestion
│   ├── 4-PGVector-search-example.py     # Vector database search
│   └── example_pdf.pdf                   # Sample PDF for testing
├── venv/                       # Virtual environment (isolated Python environment)
│   ├── bin/                    # Executables (python, pip)
│   └── lib/                    # Installed packages
├── .env.example                # Environment variables template
├── .env                        # Your actual environment variables (not in git)
├── requirements.txt            # Python dependencies (like package.json)
├── docker-compose.yml          # PostgreSQL with pgvector setup
├── init-pgvector.sql           # SQL script to enable pgvector extension
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Deactivating the Virtual Environment

When you're done:
```bash
deactivate
```

**What this does:**
- Removes virtual environment from PATH
- Returns to system Python
- Similar to:
  - Closing a project in an IDE
  - Switching Node versions
  - Stopping a Docker container (the venv directory and packages persist on disk, just like Docker volumes)

## Common Python Commands (for your reference)

```bash
# Run Python file
python filename.py

# Install package
pip install package_name

# Install from requirements.txt
pip install -r requirements.txt

# List installed packages
pip list

# Python interactive shell (REPL)
python
# or
python -i

# Check Python version
python --version
```

## Notes

- Some examples use `gpt-5-mini` model. Make sure your OpenAI API key has access to this model or update the model name in the code.
- The project uses LangChain Expression Language (LCEL) with the pipe operator (`|`) for chain composition.
- Python's dynamic typing means you won't get compile-time errors like in Java/Kotlin, but type hints help with IDE autocomplete and documentation.
- The `load_dotenv()` function in each file loads environment variables from `.env` into `os.environ`, making them accessible via `os.getenv()`.

## Troubleshooting

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

**API key errors:**
- Check `.env` file exists and has valid keys
- Ensure `.env` is in the project root
- Verify keys have proper permissions

**Python version issues:**
- Check version: `python --version`
- Some libraries may require specific Python versions
