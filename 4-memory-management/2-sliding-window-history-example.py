from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# Create in-memory chat history for each session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create a prompt with message history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Prepare inputs function that trims history with sliding window
def prepare_inputs(inputs: dict) -> dict:
    """Prepare inputs by trimming history with sliding window"""
    session_id = inputs["session_id"]
    history = get_session_history(session_id)
    
    # Trim messages to keep only last 6 (sliding window)
    trimmed_history = trim_messages(
        history.messages,
        max_tokens=6,      # Keep only last 6 messages (using len as counter)
        strategy="last",   # Keep the most recent messages
        token_counter=len, # Count messages instead of tokens
        start_on="human",  # Ensure history starts with human message
        include_system=True  # Keep system message if present
    )
    
    return {
        "input": inputs["input"],
        "history": trimmed_history
    }

# Create the chain
chain = prompt | llm

# Wrap with RunnableLambda for preparing inputs
prepare = RunnableLambda(prepare_inputs)
chain_with_prepare = prepare | chain

# Wrap with RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    chain_with_prepare,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Test the chat with trim_messages (sliding window)
print("Conversation with trim_messages (max 6 messages):")
print("-" * 50)

config = {"configurable": {"session_id": "session1"}}

# Add many messages to test the sliding window
messages = [
    "My name is Alice",
    "I live in Brazil",
    "I work as a developer",
    "I like programming",
    "I know Python",
    "I'm learning LangChain",
    "What's my name?",  # This should still work (within window)
    "Where do I live?",  # This should still work (within window)
]

for i, msg in enumerate(messages):
    # Invoke chain with the input and session_id
    response = chain_with_history.invoke(
        {"input": msg, "session_id": "session1"},
        config=config
    )
    
    print(f"User: {msg}")
    print(f"AI: {response.content}")
    print(f"  (History size: {len(get_session_history('session1').messages)} messages)")
    print()

# Show the final conversation history
print("Final conversation history:")
print("-" * 50)
history = get_session_history("session1")
for i, msg in enumerate(history.messages):
    msg_type = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f"{msg_type}: {msg.content}")
