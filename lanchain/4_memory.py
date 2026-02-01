from typing import Dict
from langchain_openai import ChatOpenAI
from config import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

def conversation_buffer_memory_test():
    # 1. 初始化llm
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )

    # 2. 构建提示词模板
    prompt = ChatPromptTemplate.from_messages([("system","你是一个友好的助手"),
                                      MessagesPlaceholder(variable_name="chat_history"),
                                      ("human", "{input}")
                                      ])

    # 3. 构建基础链
    base_chain = prompt | llm

    # 4. 为不同会话提供独立的历史存储（这里使用内存字典，生产环境需替换为数据库）
    store: Dict[str, ChatMessageHistory] = {}
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 5. 用 RunnableWithMessageHistory 包装基础链
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,  # 提供历史获取函数
        input_messages_key="input",  # 用户输入对应的键
        history_messages_key="chat_history",  # 提示词中历史占位符的变量名
    )

    response1 = chain_with_history.invoke(
        {"input": "你好，我叫张三"},
        config={"configurable": {"session_id": "user_123"}}
    )
    print(response1.content)

    # 第三次调用，模型能记住上下文
    response2 = chain_with_history.invoke(
        {"input": "1+1等于多少"},
        config={"configurable": {"session_id": "user_123"}}
    )
    print(response2.content)

    # 第三次调用，模型能记住上下文
    response3 = chain_with_history.invoke(
        {"input": "我刚才说我叫什么名字？"},
        config={"configurable": {"session_id": "user_123"}}
    )
    print(response3.content)

if __name__ == '__main__':
    conversation_buffer_memory_test()