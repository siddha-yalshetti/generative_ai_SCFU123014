from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    evidence: str


class ExtractedAnswers(BaseModel):
    answers: list[QuestionAnswer]

class SkillEvaluation(BaseModel):
    skill: str
    rating: str = Field(
        description="Strong, Adequate, Weak, or No evidence"
    )
    evidence: str
    concern: str


class SkillEvaluations(BaseModel):
    evaluations: list[SkillEvaluation]


class SkillAssessment(BaseModel):
    skill: str
    rating: str
    evidence_summary: str


class HiringScorecard(BaseModel):
    overall_recommendation: str
    confidence: str
    skill_assessment: list[SkillAssessment]
    key_strengths: list[str]
    key_concerns: list[str]
    follow_up_questions: list[str]

parser1 = PydanticOutputParser(
    pydantic_object=ExtractedAnswers
)


extract_answer_prompt = PromptTemplate.from_template(
    """
    You are an interview transcript analyst.

    Read the transcript below and extract the candidate's answer
    to each interview question.

    Preserve the candidate's meaning.

    Rules:
    - Extract only information stated by the candidate.
    - Do not infer skills, intent, seniority, or results.
    - If a question was not answered, write "Not answered".
    - Keep answers separate.
    - Include a short evidence quote when available.

    {format_instructions}

    Transcript:
    {transcript_text}
    """
).partial(
    format_instructions=parser1.get_format_instructions()
)


parser2 = PydanticOutputParser(
    pydantic_object=SkillEvaluations
)


evaluate_answers_prompt = PromptTemplate.from_template(
    """
    You are a structured interview evaluator.

    Assess the extracted answers against the requested skills.

    For every skill, use exactly one rating:

    Strong
    Adequate
    Weak
    No evidence

    Rules:
    - Evaluate only demonstrated evidence.
    - Do not make assumptions.
    - Do not reward a skill without clear evidence.
    - Distinguish Weak from No evidence.
    - Do not introduce new skills.

    {format_instructions}

    Extracted answers:
    {answers}

    Skills to assess:
    {skills_to_assess}
    """
).partial(
    format_instructions=parser2.get_format_instructions()
)

parser3 = PydanticOutputParser(
    pydantic_object=HiringScorecard
)


format_output_prompt = PromptTemplate.from_template(
    """
    You are a hiring scorecard editor.

    Convert the evaluation into a concise and neutral hiring scorecard.

    Preserve every skill rating and its evidence.

    Recommendation must be one of:

    Strong hire
    Hire
    Mixed evidence
    Do not recommend

    Confidence must be one of:

    High
    Medium
    Low

    Rules:
    - Strong hire = consistently strong evidence.
    - Hire = mostly adequate evidence with no critical gaps.
    - Mixed evidence = important skills are weak or unclear.
    - Do not recommend = clear critical weaknesses.
    - Use lower confidence when answers are vague or unanswered.

    {format_instructions}

    Evaluation:
    {evaluation}
    """
).partial(
    format_instructions=parser3.get_format_instructions()
)


extract_chain = extract_answer_prompt | model | parser1
evaluate_chain = evaluate_answers_prompt | model | parser2
format_chain = format_output_prompt | model | parser3


transcript_text = input(
    "Enter the transcript text: "
).strip()

skills_to_assess = input(
    "Enter skills to assess (comma-separated): "
).strip()

extracted_answers = extract_chain.invoke({
    "transcript_text": transcript_text
})

evaluation = evaluate_chain.invoke({
    "answers": extracted_answers.model_dump_json(),
    "skills_to_assess": skills_to_assess
})

scorecard = format_chain.invoke({
    "evaluation": evaluation.model_dump_json()
})

print("\nStep 1 - Extracted answers:")
print(extracted_answers)
print("\nStep 2 - Skills evaluation:")
print(evaluation)
print("\nStep 3 - Hiring scorecard:")
print(scorecard)