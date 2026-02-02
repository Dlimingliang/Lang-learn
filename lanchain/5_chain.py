from langchain_openai import ChatOpenAI
from config import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableBranch

def chain_test():
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
        a company that makes {product}?"
    )
    review_prompt = prompt.format_messages(product="Queen Size Sheet Set")
    print(review_prompt)
    messages = [
        {'role': 'user', 'content': f"{review_prompt}"},
    ]

    chain = llm | prompt
    result = chain.invoke(messages)
    print(result)

# 单输入输出
def single_seq_chain_test():
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    # 第一链：生成笑话
    prompt1 = ChatPromptTemplate.from_template("讲一个关于{主题}的笑话")
    chain1 = prompt1 | llm | StrOutputParser()

    # 第二链：翻译笑话
    prompt2 = ChatPromptTemplate.from_template("将下面的文本翻译成英文：{joke}")
    chain2 = prompt2 | llm | StrOutputParser()

    # 使用管道符连接两个链，前一个链的输出就是后一个链的输入
    sequential_chain = chain1 | chain2

    result = sequential_chain.invoke({"主题": "程序员"})
    print(result)  # 输出：一个关于程序员的英文笑话

# 多输入输出
def seq_chain_test():
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )

    # 第一步：生成描述 (并保留所有初始输入)
    description_prompt = ChatPromptTemplate.from_template(
        "根据商品名：{product_name} 和特点：{features}，写一份详细描述。"
    )
    generate_description = description_prompt | llm | StrOutputParser()

    # 第二步：生成邮件 (使用初始输入 + 上一步生成的描述)
    email_prompt = ChatPromptTemplate.from_template(
        "根据商品名：{product_name} 和以下描述：{description}，写一篇营销邮件。"
    )
    generate_email = email_prompt | llm | StrOutputParser()

    chain = (RunnablePassthrough.assign(description=generate_description))|generate_email
    result = chain.invoke({
        "product_name":"AI智能音箱",
        "features": "语音助手，家居控制，高保真音质"
    })
    print(result)
    print(type(result))

def route_to_physics(input_dict):
    """判断是否路由到物理链"""
    query = input_dict.get("query", "").lower()
    physics_keywords = ["力", "运动", "能量", "量子", "物理"]
    return any(keyword in query for keyword in physics_keywords)

def route_to_math(input_dict):
    """判断是否路由到数学链"""
    query = input_dict.get("query", "").lower()
    math_keywords = ["方程", "积分", "几何", "代数", "数学", "计算"]
    return any(keyword in query for keyword in math_keywords)

def route_chain_test():
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    parser = StrOutputParser()

    # 1. 物理链
    physics_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位物理学家，用严谨的物理学术语回答问题。"),
        ("user", "{query}")
    ])
    physics_chain = physics_prompt | llm | parser

    # 2. 数学链
    math_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位数学家，专注于提供精确的数学推导和公式。"),
        ("user", "{query}")
    ])
    math_chain = math_prompt | llm | parser

    # 3. 默认通用链
    default_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的AI助手。"),
        ("user", "{query}")
    ])
    default_chain = default_prompt | llm | parser

    branch = RunnableBranch(
        (route_to_physics, physics_chain),
        (route_to_math, math_chain),
        default_chain
    )

    # 现在 `branch` 就是一个可以调用的路由链
    result = branch.invoke({"query": "请解释一下牛顿第二定律。"})
    print(f"路由结果（物理）: {result}")

def llm_route_chain_test():
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    parser = StrOutputParser()

    # 1. 创建一个LLM来辅助分类
    classifier_prompt = ChatPromptTemplate.from_template("""
    请将以下问题分类到 'physics', 'math' 或 'general' 中的一个。
    只返回分类名称。

    问题：{query}
    分类：""")
    classifier_chain = classifier_prompt | llm | StrOutputParser()

    def llm_base_router(classification):
        def router(query):
            actual_class = classifier_chain.invoke(query)
            return  actual_class.strip() == classification
        return router

    # 1. 物理链
    physics_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位物理学家，用严谨的物理学术语回答问题。"),
        ("user", "{query}")
    ])
    physics_chain = physics_prompt | llm | parser

    # 2. 数学链
    math_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位数学家，专注于提供精确的数学推导和公式。"),
        ("user", "{query}")
    ])
    math_chain = math_prompt | llm | parser

    # 3. 默认通用链
    default_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人的AI助手。"),
        ("user", "{query}")
    ])
    default_chain = default_prompt | llm | parser

    branch = RunnableBranch(
        (llm_base_router("physics"), physics_chain),
        (llm_base_router("math"), math_chain),
        default_chain
    )

    # 现在 `branch` 就是一个可以调用的路由链
    result = branch.invoke({"query": "请解释一下牛顿第二定律。"})
    print(f"结果: {result}")


if __name__ == '__main__':
    # chain_test()
    # single_seq_chain_test()
    #seq_chain_test()
    #route_chain_test()
    llm_route_chain_test()