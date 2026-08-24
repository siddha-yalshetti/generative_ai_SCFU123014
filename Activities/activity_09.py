from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

history=[]

while(True):
    prompt = input("Enter Prompt...\n")
    message ={
        "role":"user",
        "content":prompt
    }
    model = "openai/gpt-oss-20b"
    history.append(message)
    if prompt.lower() == "exit":
        print("Exiting chat loop.")
        break
    if prompt.lower() == "show":
        print(history)
        break

    chat_completion = client.chat.completions.create(messages=history, model=model)
    response = chat_completion.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    print(response)