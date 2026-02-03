from config import config

from pydantic import BaseModel, Field
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool

class WeatherSearch(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(description="地域")
    unit: str = Field(description="天气的单位 默认为摄氏度")

def get_current_weather(location, unit="摄氏度"):
    weather_info = {
        "location": location,
        "temperature": "2-7",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }
    return json.dumps(weather_info)

def langchain_function_call_test():
    # 1. 创建llm
    model = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )

    # 2. 构建提示词模板
    prompt = ChatPromptTemplate.from_messages([("system", "你是一个友好的助手"),
                                               ("human", "{input}")
                                               ])
    #print(f"prompt: {prompt}")

    # 3. 绑定函数
    weather_function = convert_to_openai_tool(WeatherSearch)
    #print(f"weather_function: {weather_function}")
    model_with_function = model.bind(tools=[weather_function])

    chain = prompt | model_with_function

    # 4. 调用大模型
    # response 是 AIMessage 对象，包含结构化数据
    text = "大连天气怎么样?"
    print(f"用户提问:{text}")
    response = chain.invoke({"input": text})
    #print(f"response: {response}")


    # 5. 从大模型中读取函数并且调用
    if response.tool_calls:
        for tool_call in response.tool_calls:
            #print(f"name: {tool_call['name']}")
            #print(f"args: {tool_call['args']}")
            #print(f"id: {tool_call.get('id', 'N/A')}")

            # 执行函数调用
            if tool_call['name'] == 'WeatherSearch':
                args = tool_call['args']
                weather_result = get_current_weather(
                    location=args.get('location'),
                    unit=args.get('unit', '摄氏度')
                )

                # 6. 将返回结果一起扔给大模型
                from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
                # 构建包含工具调用结果的消息
                messages = [
                    HumanMessage(content=text),
                    response,  # 包含 tool_calls 的 AI 响应
                    ToolMessage(
                        content=weather_result,
                        tool_call_id=tool_call.get('id', '')
                    )
                ]
                
                # 再次调用模型获取最终回复
                final_response = model_with_function.invoke(messages)
                print(f"\nAI回答: {final_response.content}")

if __name__ == "__main__":
    langchain_function_call_test()


    