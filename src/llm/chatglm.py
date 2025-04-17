from langchain_openai import ChatOpenAI

from config import API_KEY_GLM, LLM_MODEL, LLM_API_BASE


def get_LLM():
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        openai_api_base= LLM_API_BASE,
        openai_api_key=API_KEY_GLM,
        streaming=False,
    )

    return llm


if __name__ == "__main__":
    llm = ChatOpenAI(
        model_name="glm-4-plus",
        openai_api_base= "https://open.bigmodel.cn/api/paas/v4/",
        openai_api_key="4ccfe9c199d147628bd5a8d18bb26f5f.TyygD9D0cAMwAZ2R",
        streaming=False,
    )
    res= llm.invoke([{"role": "user", "content": "请介绍一下自己"}]).content
    print(res)