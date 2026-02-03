from config import config
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from typing import Optional, List


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")

def tagging_test():
    # 1.创建llm
    model = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )

    # 2.创建提示词
    prompt = ChatPromptTemplate.from_messages(([
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}")
    ]))

    # 3.创建及绑定函数 - 使用 bind_tools 方法，并强制调用该工具
    tagging_model = model.bind_tools([Tagging], tool_choice="Tagging")

    # 4. 调用模型
    tagging_chain = prompt | tagging_model | JsonOutputToolsParser()
    response = tagging_chain.invoke({"input": "non mi piace questo cibo"})
    print(response)

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")
class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")

def extraction_test():
    model = ChatOpenAI(
        model=config["model"],
        temperature=0,
        base_url=config["baseUrl"],  # 你的baseUrl
        api_key=config["apiKey"],  # 你的apiKey
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
        ("human", "{input}")
    ])

    extraction_model = model.bind_tools([Information], tool_choice="Information")
    extraction_chain = prompt | extraction_model | JsonOutputToolsParser()
    response = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
    print("response:", response)
    
    # JsonOutputKeyToolsParser 的 key_name 是用来过滤工具名称，不是字段名
    # key_name="Information" 表示只返回 Information 工具的结果
    key_extraction_chain = prompt | extraction_model | JsonOutputKeyToolsParser(key_name="Information", first_tool_only=True)
    response2 = key_extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
    print("response2:", response2)
    print(type(response2))
    
    # 如果想获取 people 字段，需要从结果中提取
    if response2:
        print("people:", response2.get("people"))


if __name__ == '__main__':
    # tagging_test()
    extraction_test()
