from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()


@tool
def celsiusToFahrenheit(celsius : float) -> float:
    """This Function converts Celsius into Fahrenheit"""
    fahrenheit = (celsius*(9/5))+32
    return fahrenheit

@tool
def kilometerTomiles(kilometer : float) -> float:
    """This Function converts Kilometer into Miles"""
    miles = kilometer * 0.621371
    return miles


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

model_with_tools = model.bind_tools([celsiusToFahrenheit,kilometerTomiles])

user_query = "Convert 1km into miles"
ai_response = model_with_tools.invoke(user_query)

print("Model response:")
print(ai_response)

if ai_response.tool_calls:
    print("\nTool calls made:")
    for tool_call in ai_response.tool_calls:
        print(tool_call)

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "celsiusToFahrenheit":
            result = celsiusToFahrenheit.invoke(tool_args)
        elif tool_name == "kilometerTomiles":
            result = kilometerTomiles.invoke(tool_args)
        else:
            result = "Unknown tool"

        print("\nTool result:")
        print(result)
else:
    print("\nNo tool call was made by the model.")
    print(ai_response.content)
