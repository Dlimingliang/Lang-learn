import requests
from config import config

class LLMClient:
    def __init__(self, modeType=None, stream=False):
        self.baseUrl = config["baseUrl"]
        self.modelType = modeType or config["model"]
        self.stream = stream
        self.apiKey = config["apiKey"]
        self.headers: dict[str, str] = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.apiKey}'
        }

    def chat(self, messages: list[dict[str, str]]):
        url = f"{self.baseUrl}/chat/completions"

        payload = {
            "model": self.modelType,
            "messages": messages,
            "stream": self.stream,
            "temperature": 0,
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM请求失败: {e}")

    def get_text_content(self, response) -> str:
            """
            从LLM响应中提取纯文字内容

            Args:
                response: LLM的完整响应对象

            Returns:
                提取的纯文字内容
            """
            try:
                # 标准的OpenAI API响应格式
                if 'choices' in response and len(response['choices']) > 0:
                    choice = response['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content'].strip()
                    elif 'text' in choice:
                        return choice['text'].strip()

                # 如果是其他格式，尝试常见的字段
                if 'content' in response:
                    return response['content'].strip()
                if 'text' in response:
                    return response['text'].strip()
                if 'output' in response:
                    return response['output'].strip()

                # 如果都找不到，返回整个响应的字符串形式
                return str(response)

            except Exception as e:
                raise Exception(f"提取文字内容失败: {e}")

if __name__ == '__main__':
    client = LLMClient()
    # 发送聊天请求
    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse,\
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """
    style = """American English \
    in a calm and respectful tone
    """
    prompt = f"""Translate the text \
        that is delimited by triple backticks 
        into a style that is {style}.
        text: ```{customer_email}```
        """
    messages = [
        {'role': 'user', 'content': prompt}
    ]

    try:
        # 方法1: 获取完整响应，然后提取文字
        response = client.chat(messages)
        print(f"完整响应: {response}")
        text_content = client.get_text_content(response)
        print(f"提取的文字内容: {text_content}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()