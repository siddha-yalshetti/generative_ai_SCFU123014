# from langchain_core.tools import tool


# @tool
# def multiply(a: int, b: int):
#     """Multiply two numbers."""
#     return a * b


# @tool
# def addition(a: int, b: int):
#     """Add two numbers."""
#     return a + b


# result = multiply.invoke({
#     "a": 10,
#     "b": 5
# })


# result1 = addition.invoke({
#     "a": 10,    
#     "b": 5
# })
# print(result)
# print(result1)



from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


@tool
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b


# Give the tool to the LLM
model_with_tool = model.bind_tools([multiply])


# Ask the LLM
result = model_with_tool.invoke(
    "What is 25 multiplied by 10?"
)

print(result)