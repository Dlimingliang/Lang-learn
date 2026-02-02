from langchain_core.tools import tool
from config import config
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

def agent_test():
    # 初始化llm
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )

    # 解析工具
    @tool
    def multiply(first: int, second: int) -> int:
        "将两个整数相乘。"
        return first * second

    @tool
    def add(first_int: int, second_int: int) -> int:
        "将两个整数相加。"
        return first_int + second_int

    @tool
    def exponentiate(base: int, exponent: int) -> int:
        "求底数的幂次方。"
        return base ** exponent

    tools = [multiply, add, exponentiate]

    # 构建agent
    testAgent = create_agent(llm, tools=[multiply, add, exponentiate])

    result = testAgent.invoke(
        {"messages": [{"role": "user", "content": "1+2的和是多少"}]}
    )
    # 获取大模型的返回信息
    # result["messages"] 是消息列表，最后一条消息是AI的最终回复
    ai_message = result["messages"][-1]
    print("AI回复内容:", ai_message.content)

if __name__ == "__main__":
    agent_test()