## Exercise 3 — Meeting Transcript to Action Items Pipeline

# Build a 3-step pipeline. Step 1 extracts what was discussed from a raw meeting transcript. Step 2 identifies action items from that discussion, flagging anything where an owner or deadline is missing. Step 3 formats the final list into a structured task table.

# **Input:** `transcript_text`

# **Output:** the discussion summary from Step 1, the flagged action items from Step 2, and the final structured task table from Step 3

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

TRANSCRIPT_EXTRACTOR_PROMPT = """
You are an expert meeting analyst.

Your task is to read the raw meeting transcript and produce a concise but complete discussion summary.

Instructions:
- Identify the main topic(s) discussed.
- Summarize the key decisions, agreements, and points raised.
- Note unresolved issues, blockers, or open questions.
- Keep the output clear and professional.
- Do not invent facts that are not present in the transcript.

Return the output in this format:
1. Meeting Overview
2. Key Discussion Points
3. Decisions / Agreements
4. Open Issues / Questions

Transcript:
{transcript_text}
"""

ACTION_ITEM_IDENTIFICATION_PROMPT = """
You are an expert project coordinator.

Using the discussion summary above, identify all action items mentioned in the meeting.

Instructions:
- Extract tasks that require follow-up or completion.
- For each action item, include:
  - task
  - owner
  - deadline
  - status
  - notes
- Flag any action item where the owner is missing or the deadline is missing.
- If both owner and deadline are missing, clearly mark as "high priority gap".
- Do not include tasks that are only general discussion points without actual follow-up responsibility.

Return the result in this format:
- Task: ...
- Owner: ...
- Deadline: ...
- Status: ...
- Missing Info: ...
- Notes: ...

If no action items are found, say: "No action items identified."

Discussion Summary:
{discussion_summary}
"""

FORMATTER_PROMPT = """
You are an expert assistant that converts action items into a clean task table.

Take the flagged action items and format them into a Markdown table with the following columns:
| Task | Owner | Deadline | Status | Missing Info | Notes |

Rules:
- Keep each task short and specific.
- If an owner is missing, write "Unassigned".
- If a deadline is missing, write "Not specified".
- Highlight missing information in the "Missing Info" column.
- Keep the output as a table only, with no extra explanatory text.

Action Items:
{action_items}
"""

transcript_text = input("Enter Transcript : ")

MODEL = "openai/gpt-oss-20b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

summary = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": TRANSCRIPT_EXTRACTOR_PROMPT.format(transcript_text=transcript_text)}]
).choices[0].message.content

action_items = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": ACTION_ITEM_IDENTIFICATION_PROMPT.format(discussion_summary=summary)}]
).choices[0].message.content

formatted_tasks = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": FORMATTER_PROMPT.format(action_items=action_items)}]
).choices[0].message.content


print("DISCUSSION SUMMARY:\n", summary)
print("\nFLAGGED ACTION ITEMS:\n", action_items)
print("\nFORMATTED TASK TABLE:\n", formatted_tasks)