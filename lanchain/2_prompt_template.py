from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import config

class LLMClient:
    def __init__(self):
        self.baseUrl = config["baseUrl"]
        self.apiKey = config["apiKey"]
        self.client = OpenAI(api_key=self.apiKey, base_url=self.baseUrl)

    def chat(self, messages, model=None, stream=False):
        model = model or config["model"]
        response = self.client.chat.completions.create(model=model, messages=messages, stream=stream,temperature=0)
        print("Response structure:", response)
        return response

if __name__ == '__main__':

    client = LLMClient()

    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """
    review_template = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """

    # 加载模板
    reviewPromptTemplate = ChatPromptTemplate.from_template(review_template)
    reviewPrompt = reviewPromptTemplate.format_messages(text=customer_review)
    print(reviewPrompt)
    messages = [
        {'role': 'user', 'content': f"{reviewPrompt}"},
    ]
    try:
        # 获取完整响应，然后提取文字
        response = client.chat(messages)
        content = ""
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    chunk_content = choice.message.content
                    content += chunk_content  # 将内容累加到总内容中
        else:
            raise ValueError("Unexpected response structure")
        print(content)
        # 结构化响应
        print(type(content))
        # 不能获取gift
        # print(content.gift)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()