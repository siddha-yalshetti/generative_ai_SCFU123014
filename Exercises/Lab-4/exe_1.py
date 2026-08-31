import json
import os
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "meta/llama-3.1-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"

STEP_1_SYSTEM_PROMPT = """
Extract the key job requirements and return only valid JSON with these keys:
job_title, company, location, required_skills, preferred_skills, experience,
education, and responsibilities.
"""

STEP_2_SYSTEM_PROMPT = """
Write a short, friendly, personalized candidate outreach message.
Include a subject, greeting, relevant role matches, and a call to action.
"""


def create_client():
    """Create an OpenAI client configured for NVIDIA's API."""
    load_dotenv()
    return OpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=BASE_URL,
    )


def get_completion(client, system_prompt, user_prompt, response_format=None):
    """Send a chat completion request and return its text response."""
    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ],
    }
    if response_format:
        request["response_format"] = response_format

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content.strip()


def extract_requirements(client, job_posting):
	"""Extract structured requirements from a job posting."""
	requirements_text = get_completion(
		client,
		STEP_1_SYSTEM_PROMPT,
		job_posting,
		response_format={"type": "json_object"},
	)
	requirements_text = (
		requirements_text.replace("```json", "").replace("```", "").strip()
	)
	return json.loads(requirements_text)


def generate_outreach(client, requirements, candidate_profile):
    """Generate a personalized outreach message for a candidate."""
    user_prompt = (
        "Job requirements:\n"
        + json.dumps(requirements, indent=2)
        + "\n\nCandidate profile:\n"
        + candidate_profile
    )
    return get_completion(client, STEP_2_SYSTEM_PROMPT, user_prompt)


def main():
    """Run the job requirement extraction and outreach workflow."""
    client = create_client()
    job_posting = input("Enter the job posting: ")
    candidate_profile = input("Enter the candidate profile: ")
    requirements = extract_requirements(client, job_posting)
    outreach_message = generate_outreach(client, requirements, candidate_profile)

    print("\nStructured Requirements:")
    print(json.dumps(requirements, indent=2))
    print("\nPersonalized Outreach Message:")
    print(outreach_message)


if __name__ == "__main__":
    main()
