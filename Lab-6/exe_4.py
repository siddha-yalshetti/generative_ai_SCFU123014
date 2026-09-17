import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


@tool
def get_weather(city):
    """This function gives weather data like temperature and conditions based on the given city."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo = requests.get(geo_url, params={"name": city, "count": 1}).json()

    if "results" not in geo:
        return "City not found"

    latitude = geo["results"][0]["latitude"]
    longitude = geo["results"][0]["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather = requests.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m",
        },
    ).json()

    current = weather["current"]
    code = current["weather_code"]

    conditions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Cloudy",
        45: "Foggy",
        48: "Foggy",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rainy",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snowy",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Rain showers",
        82: "Heavy rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm",
        99: "Thunderstorm",
    }

    return {
        "city": city,
        "temperature": current["temperature_2m"],
        "condition": conditions.get(code, "Unknown"),
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"],
    }


RECOMMAND_OUTFIT_SYSTEM_PROMPT = """
You are a fashion and weather assistant.
Use the provided weather details to suggest an outfit.
Recommend clothing, footwear, and accessories based on temperature, humidity, and wind.
Keep the advice practical, concise, and appropriate for everyday wear.
Make sure that the output should be a short paragraph.
"""

@tool
def recommandOutfit(weather_info):
    """Recommend an outfit based on weather information."""
    messages = [
        ("system", RECOMMAND_OUTFIT_SYSTEM_PROMPT),
        (
            "human",
            f"Weather info: {weather_info}. Suggest the best outfit for the day and explain why.",
        ),
    ]
    outfit_suggestion = model.invoke(messages).content
    return outfit_suggestion


models_with_tools = model.bind_tools([get_weather,recommandOutfit])

user_query = "What should i wear in Solapur?"
response = models_with_tools.invoke(user_query)

# print("Model response:")
# print(response)

if response.tool_calls:
    # print("\nTool calls made:")
    for tool_call in response.tool_calls:
        # print(tool_call)

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "get_weather":
            result = get_weather.invoke(tool_args)
            recommandOutfitResponse = recommandOutfit.invoke({'weather_info':result})
            print(recommandOutfitResponse)
        else:
            result = "Unknown tool"
else:
    print("\nNo tool call was made by the model.")
    # print(response.content)

