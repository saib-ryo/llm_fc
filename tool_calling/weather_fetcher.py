import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim

# 天気コードを日本語に変換する辞書
WEATHER_CODE_JP = {
    0: "快晴", 1: "晴れ", 2: "一部曇り", 3: "曇り",
    45: "霧", 48: "霧氷を伴う霧",
    51: "霧雨（弱い）", 53: "霧雨（中程度）", 55: "霧雨（強い）",
    61: "雨（弱い）", 63: "雨（中程度）", 65: "雨（強い）",
    71: "雪（弱い）", 73: "雪（中程度）", 75: "雪（強い）",
    80: "にわか雨（弱い）", 81: "にわか雨（中程度）", 82: "にわか雨（強い）",
    95: "雷雨（弱～中）", 96: "雷雨とひょう（弱い）", 99: "雷雨とひょう（強い）"
}

def geocode_place(place: str):
    g = Nominatim(user_agent="weather_app")
    loc = g.geocode(place)
    if not loc:
        raise ValueError(f"場所が見つかりませんでした: {place}")
    return loc.latitude, loc.longitude

def call_api(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_daily(api_url, lat, lon, start_date, end_date, include_weathercode=True):
    daily_params = "temperature_2m_max,temperature_2m_min,precipitation_sum"
    if include_weathercode:
        daily_params += ",weathercode"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": daily_params,
        "timezone": "Asia/Tokyo",
        "start_date": start_date,
        "end_date": end_date
    }
    r = call_api(api_url, params)
    return r.get("daily", {})

def get_weather(lat, lon, start_date_str, end_date_str):
    start_dt = datetime.fromisoformat(start_date_str).date()
    end_dt   = datetime.fromisoformat(end_date_str).date()
    results = {}

    # --- JMA（4日以内） ---
    jma_end = min(start_dt + timedelta(days=3), end_dt)
    jma = fetch_daily("https://api.open-meteo.com/v1/jma", lat, lon,
                      start_dt.strftime("%Y-%m-%d"), jma_end.strftime("%Y-%m-%d"))
    for i, ds in enumerate(jma.get("time", [])):
        results[ds] = {
            "date": ds, "source": "JMA",
            "temp_max": jma["temperature_2m_max"][i],
            "temp_min": jma["temperature_2m_min"][i],
            "precipitation": jma["precipitation_sum"][i],
            "weather": WEATHER_CODE_JP.get(jma["weathercode"][i], f"不明（コード:{jma['weathercode'][i]})")
        }

    # --- Forecast（最大15日以内、今日 +14日まで） ---
    fc_end = min(start_dt + timedelta(days=14), end_dt)
    fc = fetch_daily("https://api.open-meteo.com/v1/forecast", lat, lon,
                     start_dt.strftime("%Y-%m-%d"), fc_end.strftime("%Y-%m-%d"))
    for i, ds in enumerate(fc.get("time", [])):
        if ds not in results or results[ds]["temp_max"] is None:
            results[ds] = {
                "date": ds, "source": "Forecast",
                "temp_max": fc["temperature_2m_max"][i],
                "temp_min": fc["temperature_2m_min"][i],
                "precipitation": fc["precipitation_sum"][i],
                "weather": WEATHER_CODE_JP.get(fc["weathercode"][i], f"不明（コード:{fc['weathercode'][i]})")
            }

    # --- Climate（17日以降、または欠損補完用） ---
    cl = fetch_daily("https://climate-api.open-meteo.com/v1/climate", lat, lon,
                     start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"),
                     include_weathercode=False)
    for i, ds in enumerate(cl.get("time", [])):
        if ds not in results or results[ds]["temp_max"] is None:
            results[ds] = {
                "date": ds, "source": "Climate",
                "temp_max": cl["temperature_2m_max"][i],
                "temp_min": cl["temperature_2m_min"][i],
                "precipitation": cl["precipitation_sum"][i],
                "weather": "(長期傾向のみ: weathercodeなし)"
            }

    return [results[d] for d in sorted(results.keys())]
