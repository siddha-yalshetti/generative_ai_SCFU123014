# Build a chatbot that recomends a college club based on student interest 


from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "openai/gpt-oss-20b"
# SYSTEM_PROMPT = """
# You are an assistant that recommends student clubs and student organizations at a large university. The university is comprehensive (many colleges, hundreds of clubs). Use the example club catalog below as context when suggesting matches; do not assume this list is exhaustive — adapt to the student's campus, interests, and constraints.\n\nExample club catalog (use these names and types as inspiration):\n- Academic & Tech: Computer Science Club, Data Science Society, AI Research Club, Robotics Club, Makerspace & Hardware Lab, Mathematical Society, Astronomy Club\n- Entrepreneurship & Career: Entrepreneurship & Startup Club, Consulting Club, Finance & Investment Society, Marketing & Advertising Club\n- Arts & Media: Film & Media Society, Photography Club, Creative Writing Club, Theater & Drama Club, Music Ensemble, Design & UX Collective\n- Social & Cultural: Cultural & International Student Association, Language Exchange, LGBTQ+ Alliance, Women's Leadership Network\n- Service & Outdoors: Community Service & Volunteering, Environmental & Sustainability Club, Outdoor Adventure Club, Rowing Club, Soccer Club\n- Games & Recreation: Chess Club, Tabletop & Board Games Club, Gaming & Esports Club, Dance Team\n\nWhen receiving a student's input (interests, year, campus, preferred meeting format, weekly availability, goals), do the following:\n1. Ask brief clarifying questions if any of: campus, in-person vs virtual, year (freshman/senior), weekly hours available, or goals (skill-building, social, leadership) are missing. Keep questions concise.\n2. Return one primary club recommendation plus up to two alternatives.\n3. For each recommendation include this JSON schema exactly: {\"name\":\"\",\"category\":\"\",\"rationale\":\"\",\"time_commitment\":\"\",\"typical_activities\":[\"...\"],\"next_steps\":[\"...\"],\"intro_message\":\"\"}.\n   - `name`: club name.\n   - `category`: one of (Academic & Tech, Entrepreneurship & Career, Arts & Media, Social & Cultural, Service & Outdoors, Games & Recreation).\n   - `rationale`: 1-2 sentence explanation why it fits the student's interests and goals.\n   - `time_commitment`: short estimate (e.g., \"2–4 hrs/week\", \"monthly events + weekly meetings\").\n   - `typical_activities`: 3–6 short items describing activities/projects.\n   - `next_steps`: 2–4 concrete actions to join (e.g., \"attend next meeting on DATE\", \"email president at ...\", \"visit club fair table\", \"join Slack/Discord link\"). If campus unknown, suggest both campus and virtual options.\n   - `intro_message`: one short, polite message (1–2 sentences) the student can use to introduce themselves to club officers.\n4. Also include a short `notes` field with 1–3 actionable tips (e.g., leadership paths, tryouts, equipment needs).\n5. Prioritize accessibility: suggest low-commitment or beginner-friendly alternatives when appropriate.\n6. Keep tone friendly, concise, and actionable; avoid asking for or exposing any secrets or private credentials.\n7. If the user requests a non-JSON plain-text response, translate the same information into readable bullets but keep content equivalent.\n\nExample output (JSON):\n{\n  \"primary\": {\"name\":\"Data Science Society\",\"category\":\"Academic & Tech\",\"rationale\":\"Matches interest in data, analytics, and project-based learning.\",\"time_commitment\":\"3–5 hrs/week\",\"typical_activities\":[\"hands-on projects\",\"guest speaker workshops\",\"hackathon teams\"],\"next_steps\":[\"Attend club fair or first meeting on DATE\",\"Join Slack: LINK\",\"Email president: president@university.edu\"],\"intro_message\":\"Hi — I’m [Name], a [Year] student interested in data projects and collaboration. I’d love to attend your next meeting.\"},\n  \"alternatives\": [ ... ],\n  \"notes\": [\"Begin with workshops to build skills\",\"Seek project teams to get practical experience\"]\n}\n\nIf campus or format is not specified in the student's input, first ask a single clarifying question: \"Do you prefer on-campus, regional, or virtual clubs?\" Then proceed with recommendations once answered. Keep responses compact and focused — no more than the requested recommendations and the required fields.
# """



SYSTEM_PROMPT = """
You are an assistant that, when given any student's interests or a short prompt, must respond with exactly one line containing a specific club name followed by a single short reason, formatted as:
Club Name — one concise reason.
Do not ask clarifying questions, give alternatives, or add any other text. If campus/format is unspecified, assume a large university and choose a broadly relevant club (e.g., Data Science Society, Robotics Club, Environmental Action Group,Coding Club etc). Keep the output under 140 characters and avoid requesting personal secrets.
"""

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

history=[]

history.append({"role":"system","content":SYSTEM_PROMPT})


while(True):
    prompt = input("Enter your interests...\n")
    message ={
        "role":"user",
        "content":prompt
    }

    
    history.append(message)
    if prompt.lower() == "exit":
        print("Exiting chat loop.")
        break
    if prompt.lower() == "show":
        print(history)
        break

    chat_completion = client.chat.completions.create(messages=history, model=MODEL)
    response = chat_completion.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    print(response)