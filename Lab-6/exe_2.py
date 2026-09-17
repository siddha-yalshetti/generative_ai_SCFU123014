from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool

load_dotenv()


class EMIInput(BaseModel):
    principal_amount: float = Field(..., description="Loan principal amount")
    annual_interest_rate: float = Field(..., description="Annual interest rate in percentage")
    time_period_years: float = Field(..., description="Loan period in years")


def calculate_emi(principal_amount: float, annual_interest_rate: float, time_period_years: float) -> float:
    """Calculate the EMI for a simple-interest based loan."""
    simple_interest = (principal_amount * annual_interest_rate * time_period_years) / 100
    total_amount = principal_amount + simple_interest
    emi = total_amount / (time_period_years * 12)
    return emi


emi_tool = StructuredTool.from_function(
    func=calculate_emi,
    name="calculate_emi",
    description="Calculate the EMI value for a loan given principal amount, annual interest rate, and time period in years.",
    args_schema=EMIInput,
)

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

model_with_tool = model.bind_tools([emi_tool])

question = "Calculate EMI for a principal amount of 100000 at 10% annual interest for 1 year."
response = model_with_tool.invoke(question)

# print("Model response:")
# print(response)

if response.tool_calls:
    tool_call_args = response.tool_calls[0]["args"]
    print("\nStructured tool arguments:")
    print(tool_call_args)

    result = emi_tool.invoke(tool_call_args)
    print("\nTool result:")
    print(result)

else:
    print(response.content)