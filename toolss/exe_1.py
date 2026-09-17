from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

@tool
def order_status(order_id: str):
    """Check the status of an order."""

    orders = {
        "ORD-45217": "Shipped",
        "ORD-45218": "Preparing",
        "ORD-45219": "Delivered"
    }
    return orders.get(order_id, "Order not found")
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

model = model.bind_tools([order_status])
question = "Where is my order ORD-45217?"
response = model.invoke(question)

response = model.invoke(question)

print("Tool Call:")
print(response.tool_calls)

if response.tool_calls:
    args = response.tool_calls[0]["args"]
    result = order_status.invoke(args)

    print("Tool Result:")
    print(result)