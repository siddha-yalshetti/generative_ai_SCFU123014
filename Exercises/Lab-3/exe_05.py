## Exercise 5 — Multi-language Translator

# Build a prompt that translates a sentence into a target language at a given formality level.

# **Input:** `sentence`, `target_language`, `formality`

# **Output:** a translated sentence matching the requested formality level



from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """
You are an expert multilingual translator.

Translate the user's sentence into {target_language}.
Use a formality level of {formality}, where 0 is very informal and 1 is very formal.

Rules:
- Preserve the original meaning, intent, and important details.
- Use natural grammar, vocabulary, and idioms for the target language.
- Match the requested formality consistently in word choice, greetings, pronouns, and sentence structure.
- Do not add explanations, opinions, context, or information that is not in the original sentence.
- Preserve names, numbers, URLs, placeholders, and formatting unless the target language requires an established convention.
- Return only the translated sentence, without quotation marks, labels, notes, or analysis.
"""

sentence = input("Enter Sentence you want to translate : ")
target_language = input("Enter Target Language : ")
formality = input("Enter formality level (0-1) : ")

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
        "content":"Sentence"+sentence+"\n"+"Target Language :"+target_language+"\n"+"Formality Level :"+formality+"\n"
    }
]
chat_completion = client.chat.completions.create(messages=messages, model=MODEL)
response = chat_completion.choices[0].message.content
print(response)