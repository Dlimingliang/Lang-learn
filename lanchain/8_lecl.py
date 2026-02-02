from langchain_openai import ChatOpenAI
from config import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def simple_lecl_test():
    llm = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )
    prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"topic":"熊"})
    print(response)
    print(type(response))

if __name__ == '__main__':
    simple_lecl_test()