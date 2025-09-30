from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def recommend_outfit_with_llm(temp_max, temp_min, precipitation, weather):
    prompt = f"""
    以下の条件に基づいて、1日を快適に過ごすためのおすすめの服装を日本語で提案してください。

    - 最高気温: {temp_max if temp_max is not None else "不明"}℃
    - 最低気温: {temp_min if temp_min is not None else "不明"}℃
    - 天気: {weather}
    - 降水量: {precipitation if precipitation is not None else "不明"}mm

    出力条件:
    - 日本語
    - 1〜2文程度
    - 服装や持ち物（傘、防寒具など）の具体的な提案
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()
