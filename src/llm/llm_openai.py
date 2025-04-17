from langchain_openai import ChatOpenAI
from config import LLM_MODEL, BASE_URL_OPENAI, API_KEY_OPENAI

def get_ChatOpenAI():
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        base_url=BASE_URL_OPENAI,
        openai_api_key=API_KEY_OPENAI,
        streaming=True,
    )

    return llm



if __name__ == "__main__":

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        base_url="https://www.dmxapi.cn/v1",
        openai_api_key="sk-5geduntoNvm5OTAla9PDcnNnzErVD3zFSp8ZM0OxvrCPyT5o",
        streaming=True,
    )
    from langchain.schema import HumanMessage
    res = llm.invoke([HumanMessage(content="请介绍一下自己")])
    llm.predict("请问TuGraph-DB的安装步骤是什么？")
    print(res)