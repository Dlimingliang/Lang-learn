import requests
from pydantic import BaseModel, Field
import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage

# 自定义解析器：解析 OpenAI 工具调用结果
def parse_tool_output(message: AIMessage):
    """解析 AI 消息，返回 AgentAction 或 AgentFinish"""
    if message.tool_calls:
        # 有工具调用
        tool_call = message.tool_calls[0]
        return AgentAction(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
            log=str(message)
        )
    else:
        # 没有工具调用，返回最终结果
        return AgentFinish(
            return_values={"output": message.content},
            log=str(message)
        )

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""

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
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'


def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)

def route_test():
    model = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    route_chain = model.bind_tools([get_current_temperature])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        ("user", "{input}"),
    ])
    
    # 使用自定义解析器来解析输出并执行 route
    chain = prompt | route_chain | parse_tool_output | route
    # response = chain.invoke({"input": "what is the weather in sf right now"})
    response = chain.invoke({"input": "hi"})
    print(response)

if __name__ == '__main__':
    route_test()