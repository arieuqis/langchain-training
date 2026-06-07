from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

# Create the chain
chain = prompt | llm

# Wrap with RunnableWithMessageHistory
# This automatically manages message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Test the chat with history using RunnableWithMessageHistory
print("Conversation with RunnableWithMessageHistory:")
print("-" * 50)

# Reusable config for session1
config = {"configurable": {"session_id": "session1"}}

# First interaction
response1 = chain_with_history.invoke(
    {"input": "Hi, my name is Alice"},
    config=config
)
print(f"User: Hi, my name is Alice")
print(f"AI: {response1.content}")
print()

# Second interaction - remembers the name
response2 = chain_with_history.invoke(
    {"input": "What's my name?"},
    config=config
)
print(f"User: What's my name?")
print(f"AI: {response2.content}")
print()

# Third interaction - remembers previous question
response3 = chain_with_history.invoke(
    {"input": "What did I just ask?"},
    config=config
)
print(f"User: What did I just ask?")
print(f"AI: {response3.content}")
print()

# Show the full conversation history for this session
print("Full conversation history for session1:")
print("-" * 50)
history = get_session_history("session1")
for i, msg in enumerate(history.messages):
    msg_type = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f"{msg_type}: {msg.content}")
