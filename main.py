from openai import OpenAI
from dotenv import load_dotenv
import os, json, requests, math
from datetime import datetime
from difflib import SequenceMatcher

# .env 読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# Haversine で距離計算
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# ------------------------------
# ChatGPTでホテル候補を取得
# ------------------------------
def get_hotel_candidates_via_llm(hotel_name: str, location: str, limit: int = 5, threshold: float = 0.7):
    q = (
        f"次のホテル名に基づいて候補を最大{limit}件返してください。\n"
        f"ホテル名: {hotel_name}, 地域: {location}\n"
        "各候補には必ず類似度スコア(match_score: 0.0〜1.0)を含めてください。"
        "完全一致に近いものは1.0、部分一致は0.7程度、無関係なものは0.5以下にしてください。\n"
        "出力形式は必ずJSONで、以下の形式のみを返してください:\n"
        "{\n"
        "  \"candidates\": [\n"
        "    {\"name\": ホテル名, \"address\": 住所, \"lat\": 緯度(float), \"lon\": 経度(float), \"match_score\": 類似度(float)}\n"
        "  ]\n"
        "}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": q}]
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        if "candidates" not in data:
            return []
        filtered = [c for c in data["candidates"] if c.get("match_score", 0) >= threshold]
        return filtered
    except Exception:
        return []

# ------------------------------
# 重複候補をまとめる処理
# ------------------------------
def deduplicate_hotels(candidates, name_threshold=0.85, coord_threshold=0.01):
    unique = []

    def normalize(name):
        return name.lower().replace(" ", "").replace("　", "")

    for c in candidates:
        cname = normalize(c["name"])
        is_duplicate = False
        for u in unique:
            uname = normalize(u["name"])
            sim = SequenceMatcher(None, cname, uname).ratio()
            if sim >= name_threshold and abs(c["lat"] - u["lat"]) < coord_threshold and abs(c["lon"] - u["lon"]) < coord_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(c)

    return unique

# ------------------------------
# 天気取得
# ------------------------------
def get_coordinates(location: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=ja&format=json"
    resp = requests.get(url, timeout=10).json()
    if "results" in resp and len(resp["results"]) > 0:
        return {"lat": resp["results"][0]["latitude"], "lon": resp["results"][0]["longitude"]}
    return None

def get_weather(location: str, days: int = 7):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    coords = get_coordinates(location)
    if not coords:
        return {"error": "座標を取得できませんでした"}
    lat, lon = coords["lat"], coords["lon"]

    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts&units=metric&lang=ja&appid={api_key}"
    resp = requests.get(url, timeout=10).json()

    if "daily" not in resp:
        return {"error": "週間天気データを取得できませんでした"}

    forecasts = []
    for idx, d in enumerate(resp["daily"][:days]):
        dt = datetime.utcfromtimestamp(d["dt"]).strftime("%Y-%m-%d")
        forecasts.append({
            "day": f"Day {idx+1}",
            "date": dt,
            "max_temp": f"{d['temp']['max']}°C",
            "min_temp": f"{d['temp']['min']}°C",
            "condition": d["weather"][0]["description"]
        })
    return {"location": location, "forecasts": forecasts}

# ------------------------------
# 観光スポット（ChatGPTフォールバック）
# ------------------------------
def get_tourist_spots(location: str, limit: int = 12):
    q = (
        f"{location}の代表的な観光スポットと、夜に楽しめるナイトライフや地元料理を{limit}件、"
        "名前と簡単な説明をJSONで返してください。"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": q}]
    )
    try:
        return {"location": location, "spots": json.loads(resp.choices[0].message.content)}
    except Exception:
        return {"error": "観光スポット情報を取得できませんでした"}

# ------------------------------
# メイン処理
# ------------------------------
user_input = input("旅行について、場所と期間を入力してください: ")

# location, days, arrival_time, departure_time を抽出
extract = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "ユーザー入力から location（日程地）, days（日数）, arrival_time（到着日時）, departure_time（出発日時）をJSONで返してください。"},
        {"role": "user", "content": user_input}
    ]
)
info = json.loads(extract.choices[0].message.content)

if not info.get("arrival_time"):
    info["arrival_time"] = "初日 14:00"  # デフォルト到着時刻
if not info.get("departure_time"):
    info["departure_time"] = "最終日 12:00"  # デフォルト出発時刻

print("抽出情報:", info)

# ホテル候補を取得して選択
hotel_info = None
while not hotel_info:
    hotel_name = input("宿泊ホテル名を入力してください: ")
    candidates = get_hotel_candidates_via_llm(hotel_name, info["location"])
    if not candidates or len(candidates) == 0:
        print("⚠️ 十分に一致するホテル候補が見つかりませんでした。もう一度入力してください。")
        continue

    candidates = deduplicate_hotels(candidates)

    # 第1候補を基準に距離とスコアを計算
    base = candidates[0]
    for c in candidates:
        dist = haversine(base["lat"], base["lon"], c["lat"], c["lon"])
        c["distance_km"] = round(dist, 2)
        c["final_score"] = round(c["match_score"] - (dist / 20), 3)

    candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    print("\n候補リスト（類似度＋距離でソート、重複除去後）:")
    for i, c in enumerate(candidates, start=1):
        print(f"{i}. {c['name']} - {c['address']} (score: {c['match_score']}, dist: {c['distance_km']}km, final: {c['final_score']})")
    print("0. 再入力")

    choice = int(input("番号を選んでください: "))
    if choice == 0:
        continue
    if 1 <= choice <= len(candidates):
        hotel_info = candidates[choice-1]

print(f"\n✅ 選択されたホテル: {hotel_info['name']} - {hotel_info['address']}")

# 天気 & 観光
result_weather = get_weather(info["location"], days=info.get("days", 7))
result_spots = get_tourist_spots(info["location"], limit=12)

combined = {
    "weather": result_weather,
    "spots": result_spots,
    "arrival_time": info.get("arrival_time"),
    "departure_time": info.get("departure_time"),
    "hotel": hotel_info
}
print("Function result:", json.dumps(combined, ensure_ascii=False, indent=2))

# ------------------------------
# 天気を整形して渡す
# ------------------------------
weather_text = ""
if "forecasts" in result_weather:
    weather_text = f"📅 週間天気 ({result_weather.get('location','不明')}):\n"
    for f in result_weather["forecasts"]:
        weather_text += (
            f"{f['day']} ({f['date']}): "
            f"最高 {f['max_temp']} / 最低 {f['min_temp']} / 天気: {f['condition']}\n"
        )

# ------------------------------
# LLMで旅行プランを生成
# ------------------------------
followup = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "あなたは旅行プランナーです。以下の情報をもとに、Day1〜DayNの旅行プランをカレンダー形式で作成してください。"
                "各Dayの冒頭に天気情報を載せ、その気温と天気に基づいて服装アドバイスを必ず書いてください。"
                "午前・午後・夜に分けて観光やアクティビティを提案してください。"
                "夜はナイトライフや夜景に加えて、その土地の代表的な地元料理を日ごとに違うものを提案してください。"
                "初日は到着時刻を考慮し、それ以前は活動を入れないでください。"
                "最終日は出発時刻を考慮し、搭乗1時間前には空港チェックインを行う必要があるため、その時間以降は活動を入れないでください。"
                "宿泊ホテル情報がある場合、各日の最初に『ホテル出発』、最後に『ホテルに戻る』を必ず含めてください。"
                "最後に全体の持ち物リストをまとめてください。"
            ),
        },
        {"role": "user", "content": f"旅行リクエスト: {user_input}\n\n{weather_text}\n宿泊ホテル: {hotel_info['name']} ({hotel_info['address']})"},
        {"role": "function", "name": "get_travel_info", "content": json.dumps(combined, ensure_ascii=False)},
    ]
)

print("\n💡 旅行プラン回答:\n", followup.choices[0].message.content)
