from config import EVAL_DATASET_PATH, EMBEDDING_MODEL_NAME, loglog
from rag_call import answer
import json
from sentence_transformers import SentenceTransformer, util


def answer_evaluation(eval_data: list[dict]) -> list[dict]:
    # 加载预训练模型
    model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME)
    res = []

    for data in eval_data:
        loglog.debug(f"正在评估问题: {data['question']}, id: {data['id']}")
        question = data["question"]
        answer_result = answer(question)

        # 将句子转换为句子嵌入
        embedding1 = model.encode(data["answer"], convert_to_tensor=True)
        embedding2 = model.encode(answer_result, convert_to_tensor=True)
        # 计算余弦相似度
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

        res.append({
            "id": data["id"],
            "question": question,
            "answer": answer_result,
            "cosine_score": (cosine_scores.item() + 1) / 2,  # 将余弦相似度转换为0-1之间的值
        })
    return res


def evaluate():
    """
    评估函数：
    1. 读取评估问题集 调用rag系统回答并保存结果至文件
    2. 读取评估结果文件和评估答案集，计算相似度 取平均
    """
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    # 评估
    result = answer_evaluation(eval_data)
    # 保存结果
    with open("../data/bge-large-zh-v1.5.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    # 计算平均相似度
    avg_score = sum([item["cosine_score"] for item in result]) / len(result)
    print(f"平均相似度: {avg_score:.4f}")


