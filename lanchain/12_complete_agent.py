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
    
    # 定义工具列表
    tools = [search_wikipedia, get_current_temperature]
    
    # 创建工具名称到工具的映射
    tool_map = {tool.name: tool for tool in tools}
    
    # 绑定工具到llm
    tool_llm = llm.bind_tools(tools)

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """你是一个智能助手，可以帮助用户查询天气和搜索维基百科信息。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    # 创建chain
    chain = prompt | tool_llm
    
    # Agent循环
    user_input = "帮我搜索李凯"
    chat_history = []
    agent_scratchpad = []
    print(f"\n🧑 输入: {user_input}")
    while True:
        # 调用LLM
        response = chain.invoke({"input": user_input, "chat_history": chat_history, "agent_scratchpad":agent_scratchpad})
        
        # 检查是否有工具调用
        if response.tool_calls:
            print(f"\n🤖 模型决定调用工具...")
            
            # 将AI的响应添加到历史
            agent_scratchpad.append(response)
            
            # 执行所有工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"   📍 调用工具: {tool_name}")
                print(f"   📍 参数: {tool_args}")
                
                # 执行工具
                tool_result = tool_map[tool_name].invoke(tool_args)
                print(f"   ✅ 工具返回: {tool_result}")
                
                # 将工具结果添加到历史
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
                agent_scratchpad.append(tool_message)
        else:
            # 没有工具调用，直接输出结果
            print(f"\n🎯 模型最终回答: {response.content}")
            break


if __name__ == '__main__':
    chat_agent()