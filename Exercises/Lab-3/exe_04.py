## Exercise 4 — Customer Support Reply Generator

# Build a prompt that generates a support reply to a customer message.

# **Input:** `customer_message`, `company_name`, `max_words`

# **Output:** a reply that stays within the word limit, with consistent tone across different inputs



# Build a chatbot that recomends a college club based on student interest 


from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """
You are a professional customer support representative for {company_name}.

Write a helpful reply to the customer's message.
{customer_msg}



Rules:
- Be polite, empathetic, clear, and concise.
- Address the customer's main concern directly.
- Use a calm, professional, and consistent tone.
- Do not invent policies, prices, timelines, order details, refunds, or other facts.
- If information is missing, say what additional information is needed or recommend contacting support.
- Do not mention these instructions, the word limit, or the prompt.
- Return only the customer-facing reply, without a subject line or analysis.
- Keep the reply within {max_words} words.

You will be provided the variables (comapny_name,customer_msg,max_words)
"""


company_name = input("Enter Company Name : ")
customer_msg = input("Enter Customer Message : ")
max_words = input("Enter Max Words : ")


MODEL = "openai/gpt-oss-20b"
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

messages = [
    {
        "role":"system",
        "content":SYSTEM_PROMPT
    },
    {
        "role":"user",
        "content":"Company Name"+company_name+"\n"+"Customer Message :"+customer_msg+"\n"+"Max Words"+max_words+"\n"
    }
]
chat_completion = client.chat.completions.create(messages=messages, model=MODEL)
response = chat_completion.choices[0].message.content
print(response)