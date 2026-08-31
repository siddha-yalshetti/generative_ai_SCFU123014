import json
import os

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "meta/llama-3.1-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"

STEP_1_SYSTEM_PROMPT = """
Extract the core factual claims from the news article.
Return only valid JSON in this format:
{
	"claims": [
		"claim 1",
		"claim 2"
	]
}
Include only claims supported by the article. Do not add opinions or outside facts.
"""

STEP_2_SYSTEM_PROMPT = """
Create a short fact card using only the extracted claims provided by the user.
Return only valid JSON in this format:
{
	"headline": "short factual headline",
	"bullet_points": ["point 1", "point 2", "point 3"],
	"source_confidence_note": "brief note about confidence based on the claims"
}
Use exactly three bullet points. Do not add facts that are not in the claims.
"""


def create_client():
	"""Create an OpenAI client configured for NVIDIA's API."""
	load_dotenv()
	return OpenAI(
		api_key=os.getenv("NVIDIA_API_KEY"),
		base_url=BASE_URL,
	)


def get_completion(client, system_prompt, user_prompt):
	"""Send a JSON chat completion request and return its text response."""
	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": system_prompt.strip()},
			{"role": "user", "content": user_prompt},
		],
		response_format={"type": "json_object"},
	)
	return response.choices[0].message.content.strip()


def parse_json_response(response_text):
	"""Parse a JSON response, including responses wrapped in a code block."""
	cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
	return json.loads(cleaned_text)


def extract_claims(client, article):
	"""Extract core claims from a news article."""
	response_text = get_completion(client, STEP_1_SYSTEM_PROMPT, article)
	return parse_json_response(response_text)


def generate_fact_card(client, claims):
	"""Generate a fact card using only the extracted claims."""
	claims_prompt = (
		"Use only these extracted claims to create the fact card:\n"
		+ json.dumps(claims, indent=2)
	)
	response_text = get_completion(client, STEP_2_SYSTEM_PROMPT, claims_prompt)
	return parse_json_response(response_text)


def main():
	"""Run the news article to fact card pipeline."""
	client = create_client()
	article = input("Enter the news article: ")

	extracted_claims = extract_claims(client, article)
	fact_card = generate_fact_card(client, extracted_claims)

	print("\nExtracted Claims:")
	print(json.dumps(extracted_claims, indent=2))
	print("\nFact Card:")
	print(json.dumps(fact_card, indent=2))


if __name__ == "__main__":
	main()
