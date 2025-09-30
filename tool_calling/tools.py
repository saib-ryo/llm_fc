import json
from weather_fetcher import geocode_place, get_weather
from outfit_recommender import recommend_outfit_with_llm

# -------- Python 側の実処理 -------- #
def fetch_weather_tool(place: str, start_date: str, end_date: str):
    """天気情報取得処理"""
    lat, lon = geocode_place(place)
    return get_weather(lat, lon, start_date, end_date)

def recommend_outfit_tool(temp_max: float, temp_min: float, precipitation: float, weather: str):
    """服装提案処理"""
    return recommend_outfit_with_llm(temp_max, temp_min, precipitation, weather)


# -------- LLM に渡す tool 定義 -------- #
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "description": "指定された場所と期間の天気情報を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"}
                },
                "required": ["place", "start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_outfit",
            "description": "気温と天気に基づいて服装を提案する",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp_max": {"type": "number"},
                    "temp_min": {"type": "number"},
                    "precipitation": {"type": "number"},
                    "weather": {"type": "string"}
                },
                "required": ["temp_max", "temp_min", "precipitation", "weather"]
            }
        }
    }
]
