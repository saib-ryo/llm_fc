from openai import OpenAI
from dotenv import load_dotenv
import os, json
from tools import tools, fetch_weather_tool, recommend_outfit_tool

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ユーザー入力
user_input = input("場所と期間を入力してください（例: 東京で2025年10月1日から2025年10月20日まで）: ")

# LLM に入力を投げて tool 呼び出しを待つ
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": user_input}],
    tools=tools
)

msg = resp.choices[0].message

# --- ツール呼び出し処理 ---
if msg.tool_calls:
    for call in msg.tool_calls:
        func_name = call.function.name
        args = json.loads(call.function.arguments)

        if func_name == "fetch_weather":
            place = args["place"]
            start_date = args["start_date"]
            end_date = args["end_date"]

            rows = fetch_weather_tool(place, start_date, end_date)

            print(f"\n📍 {place} の天気予報 {start_date} ～ {end_date}\n")
            for r in rows:
                outfit = recommend_outfit_tool(
                    r["temp_max"], r["temp_min"], r["precipitation"], r["weather"]
                )
                print(f"{r['date']} [{r['source']}]: {r['weather']} / "
                      f"最高 {r['temp_max']}℃ / 最低 {r['temp_min']}℃ / 降水量 {r['precipitation']}mm")
                print(f"  👕 {outfit}\n")
else:
    print("ツール呼び出しがされませんでした。")
