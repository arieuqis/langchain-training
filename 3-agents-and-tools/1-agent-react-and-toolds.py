from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


@tool("Calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression) # be careful with eval, because it can execute any code
        return str(result)
    except Exception as e:
        return f"Error: Invalid expression - {str(e)}"

@tool("web_search")
def web_search(query: str) -> str:
    """Mock web search tool. Returns a hardcoded response."""
    data = {"Brazil": "Brasilia", "Argentina": "Buenos Aires", "Chile": "Santiago"}
    return f"Capital of {query} is {data.get(query, 'Unknown')}"
    
llm = ChatOpenAI(model="gpt-5-mini", disable_streaming=True)
tools = [calculator, web_search]

system_prompt = """You are a helpful assistant. Answer the following question using the available tools.

You have access to the following tools:
{tools}

Use the tools to answer the question. Your answers must be based on what the tools return.
You must use the tools even if you already know the answer.

Rules:
- If you choose an action, do not include final answer in the same step
- After action and action input, stop and wait for observation
- Never search the internet, only use the tools provided

Follow this format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {question}
Thought: {agent_scratchpad}"""

agent_graph = create_agent(llm, tools, system_prompt=system_prompt)

# Invoke the agent with a question
result = agent_graph.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
# Extract the final answer from the last message
final_answer = result["messages"][-1].content
print("Answer:", final_answer)

result2 = agent_graph.invoke({"messages": [{"role": "user", "content": "What is the capital of Brazil?"}]})
final_answer2 = result2["messages"][-1].content
print("Answer:", final_answer2)
