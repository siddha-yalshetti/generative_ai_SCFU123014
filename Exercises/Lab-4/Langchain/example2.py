from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import sys

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Explain {topic} in simple words.")
])
prompt_result = prompt.invoke({"topic": "RAG"})


result = model.invoke(prompt_result)

print(result.content)