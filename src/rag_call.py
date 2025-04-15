from rag import RAG
from config import loglog

def _process_result(result):
    """
    对生成结果的后处理，暂时不用
    """
    return result

# 调用RAG模型进行问答，可选择是否利用llm重写问题 
def answer(question: str, use_ensemble: bool = False, is_rewrite: bool = True):
    loglog.debug(f"正在初始化RAG对象，需要加载 re-rank模型...")
    rag = RAG(use_ensemble)
    loglog.debug(f"RAG模型初始化完成, use_ensemble = {use_ensemble}")

    result = rag.get_answer(question, is_rewrite)
    processed_result = _process_result(result)

    return processed_result



if __name__ == "__main__":
    
    question = "请问TuGraph-DB的安装步骤是什么？"
    answer_result = answer(question)
    print(answer_result)