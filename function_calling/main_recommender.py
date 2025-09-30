from weather_fetcher import geocode_place, get_weather
from outfit_recommender import recommend_outfit_with_llm
from openai import OpenAI
from dotenv import load_dotenv
import os, json, re

# === JSON抽出補助 ===
def parse_json_from_llm(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}")+1]
    return json.loads(text)

def parse_input_with_llm(user_input: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    次の文章から場所、開始日、終了日を抽出してください。
    出力はJSON形式で "place", "start_date", "end_date" にしてください。
    日付はYYYY-MM-DD形式。
    入力: {user_input}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return parse_json_from_llm(resp.choices[0].message.content)

if __name__ == "__main__":
    user_input = input("場所と期間を入力してください（例: 東京で2025年10月1日から2025年10月20日まで）: ")
    parsed = parse_input_with_llm(user_input)
    place = parsed["place"]
    start_date = parsed["start_date"]
    end_date = parsed["end_date"]

    lat, lon = geocode_place(place)
    rows = get_weather(lat, lon, start_date, end_date)

    print(f"\n📍 {place} の天気予報 {start_date} ～ {end_date}\n")
    for r in rows:
        outfit = recommend_outfit_with_llm(r["temp_max"], r["temp_min"], r["precipitation"], r["weather"])
        print(f"{r['date']} [{r['source']}]: {r['weather']} / "
            f"最高 {r['temp_max']}℃ / 最低 {r['temp_min']}℃ / 降水量 {r['precipitation']}mm")
        print(f"  👕 {outfit}\n")

