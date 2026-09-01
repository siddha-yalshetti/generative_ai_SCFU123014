from langchain_groq import ChatGroq
from dotenv import load_dotenv
import sys

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

result = model.invoke("What is the capital of India?")

print(result.content)