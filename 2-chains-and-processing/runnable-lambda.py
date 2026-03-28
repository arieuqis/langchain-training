from langchain_core.runnables import RunnableLambda

def square(x: int) -> int:
    return x * x

square_runnable = RunnableLambda(square)

print(square_runnable.invoke(5))
