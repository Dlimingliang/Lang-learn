from openai import OpenAI
from config import config
import json

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }
    return json.dumps(weather_info)

# 可用函数的映射
available_functions = {
    "get_current_weather": get_current_weather
}

def openai_function_call_test():
    # 使用 tools 参数（新版 API 格式）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston?"
        }
    ]
    
    client = OpenAI(api_key=config["apiKey"], base_url=config["baseUrl"])
    
    # 第一次调用：让模型决定是否需要调用函数
    print("=== 第一次请求 ===")
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        tools=tools,  # 使用 tools 而非 functions
        tool_choice="auto",  # 让模型自动决定是否调用函数
        temperature=0
    )

    response_message = response.choices[0].message
    print(f"模型响应: {response_message}")
    
    # 检查模型是否要调用函数
    if response_message.tool_calls:
        print("\n=== 模型请求调用函数 ===")

        # 将助手消息添加到对话历史（需要转换为字典格式）
        assistant_message = {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": str(tc.id),  # 确保是字符串类型
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response_message.tool_calls
            ]
        }
        messages.append(assistant_message)
        
        # 处理每个函数调用
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"调用函数: {function_name}")
            print(f"参数: {function_args}")
            
            # 执行函数
            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                print(f"函数返回: {function_response}")
                
                # 将函数结果添加到消息中
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(tool_call.id),  # 确保是字符串类型
                    "name": function_name,
                    "content": str(function_response)  # 确保是字符串类型
                })
        
        # 第二次调用：将函数结果发送给模型，获取最终回复
        print("\n=== 第二次请求（带函数结果）===")
        print(messages)
        second_response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=0
        )
        
        final_message = second_response.choices[0].message.content
        print(f"\n最终回复: {final_message}")
    else:
        # 模型直接回复，不需要调用函数
        print(f"\n模型直接回复: {response_message.content}")

if __name__ == '__main__':
    openai_function_call_test()
