from sentence_transformers import SentenceTransformer, util

if __name__ == "__main__":
    sen1 = "TuGraph-DB有2套http的接口，分别在src/restful/server/rest_server.cpp和src/http/http_server.cpp"
    sen2 = "TuGraph-DB有http接口，对应的接口代码在RESTful API文档中。"
    sen3 = "对不起，我不太明白你的意思。"

    model = SentenceTransformer(model_name_or_path="BAAI/bge-large-zh-v1.5")
    # model = SentenceTransformer(model_name_or_path="BAAI/bge-m3")
    embedding1 = model.encode(sen1, convert_to_tensor=True)
    embedding2 = model.encode(sen3, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    print(f"余弦相似度: {(cosine_scores.item() + 1) / 2:.4f}") 