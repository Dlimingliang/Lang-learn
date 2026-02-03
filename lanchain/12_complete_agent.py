from config import config

import requests
from pydantic import BaseModel, Field
import datetime
import wikipedia

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="获取天气数据的位置纬度")
    longitude: float = Field(..., description="获取天气数据的位置经度")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """获取给定坐标的当前温度."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"获取温度数据失败: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'当前温度为 {current_temperature}°C'

@tool
def search_wikipedia(query: str) -> str:
    """运行维基百科搜索并获取页面摘要"""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"标题: {page_title}\n摘要: {wiki_page.summary}")
            pass
    if not summaries:
        return "在维基百科中没有找到有效信息"
    return "\n\n".join(summaries)


def chat_agent():
    # 创建llm,并绑定工具
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    tool_llm = llm.bind_tools([search_wikipedia,get_current_temperature])

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个小助手"),
            ("user","{input}"),
        ]
    )


if __name__ == '__main__':
    chat_agent()