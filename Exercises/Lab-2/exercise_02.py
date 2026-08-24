## Exercise 2 — Content Generator

# Build an app that generates Instagram captions for a college fest.

# **Required functionality:**

# - Accepts two inputs: fest name and fest theme
# - Generates exactly 3 caption options, each under 30 words
# - Each caption includes at least one relevant hashtag


from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = """
You are an Instagram caption generator for college festivals.

The user will provide:
1. The fest name
2. The fest theme

Generate exactly 3 distinct Instagram caption options.

Rules:
- Each caption must be fewer than 30 words.
- Each caption must include at least one relevant hashtag.
- Make captions energetic, creative, and suitable for college students.
- Clearly label them as Caption 1, Caption 2, and Caption 3.
- Do not include explanations or extra text.
"""

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

fest_name = input("Enter your fest name: ").strip()
fest_theme = input("Enter your fest theme: ").strip()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            f"Fest name: {fest_name}\n"
            f"Fest theme: {fest_theme}\n"
            "Generate the 3 caption options now."
        ),
    },
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model=MODEL,
)
response = chat_completion.choices[0].message.content
print(response)