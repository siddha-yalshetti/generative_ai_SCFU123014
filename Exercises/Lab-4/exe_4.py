from dotenv import load_dotenv
from groq import Groq
import os
import json
import sys

# Load .env file
load_dotenv()

# Get API key
API_KEY = os.getenv("GROQ_API_KEY")

# Model
MODEL = "openai/gpt-oss-20b"

# Check API key
if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

# Create Groq client
client = Groq(api_key=API_KEY)


STRUCTURED_PITCH_PROMPT = """
You are an expert product strategist.

Turn the one-line product idea into a clear, structured pitch using only information implied by the idea.

Return valid JSON only with these exact keys:
{{
  "problem": "...",
  "solution": "...",
  "target_user": "...",
  "value_proposition": "...",
  "differentiator": "..."
}}

Rules:
- Do not invent facts beyond the idea.
- Keep each field specific and useful for product evaluation.
- Return valid JSON with double quotes.
- No markdown fences.
- No extra text before or after the JSON.

Product Idea:
{product_idea}
"""


INVESTOR_PITCH_PROMPT = """
You are an expert startup storyteller.

Using ONLY the structured pitch below, write a concise investor-style pitch paragraph.

Important:
- Do not use the original product idea text.
- Use only the structured JSON values.
- Write exactly one paragraph.
- Keep it persuasive and short.

Structured Pitch JSON:
{structured_pitch}
"""


def call_groq(prompt: str) -> str:
    """Call the Groq model with the provided prompt."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed: {exc}"
        ) from exc


def build_structured_pitch(product_idea: str) -> dict:
    """Step 1: Expand a one-line idea into a structured pitch."""

    if not product_idea or not product_idea.strip():
        raise ValueError("Product idea cannot be empty.")

    # Insert product idea into prompt
    prompt = STRUCTURED_PITCH_PROMPT.format(
        product_idea=product_idea
    )

    # Call LLM
    raw_output = call_groq(prompt)

    # Convert JSON string to Python dictionary
    try:
        structured_pitch = json.loads(raw_output)

    except json.JSONDecodeError:

        cleaned = raw_output.strip()

        if cleaned.startswith("```"):
            cleaned = (
                cleaned
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        structured_pitch = json.loads(cleaned)

    # Required fields
    required_keys = [
        "problem",
        "solution",
        "target_user",
        "value_proposition",
        "differentiator"
    ]

    for key in required_keys:
        structured_pitch.setdefault(
            key,
            "Not provided"
        )

    return structured_pitch


def generate_investor_pitch(structured_pitch: dict) -> str:
    """Step 2: Generate investor-style pitch."""

    # Convert dictionary to JSON
    structured_json = json.dumps(
        structured_pitch,
        ensure_ascii=False
    )

    # Create prompt
    prompt = INVESTOR_PITCH_PROMPT.format(
        structured_pitch=structured_json
    )

    # Call LLM
    return call_groq(prompt)


def run_pipeline(product_idea: str):
    """Run the complete 2-step pipeline."""

    # STEP 1
    structured_pitch = build_structured_pitch(
        product_idea
    )

    # STEP 2
    investor_pitch = generate_investor_pitch(
        structured_pitch
    )

    return {
        "structured_pitch": structured_pitch,
        "investor_pitch": investor_pitch,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        product_idea = " ".join(sys.argv[1:])
    else:
        product_idea = input("Enter a product idea: ").strip()

    result = run_pipeline(product_idea)
    print(json.dumps(result, ensure_ascii=False, indent=2))