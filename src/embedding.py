
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_DB_NAME, loglog
from tools import tool



def _create_db(documents, embedding):
    db = FAISS.from_documents(
        documents, 
        embedding
    )
    db.save_local(f"data/{EMBEDDING_DB_NAME}")
    return db


def init_embeddings(urls : list[str]):
    # 1.分解url网页内容
    urls = tuple(urls)
    print("正在分解url网页内容...")
    documents = tool.get_contents(urls)
    loglog.debug(f"分解url网页内容完成, 当前文档数量: len(documents) = {len(documents)}")

    # 2.embedding模型
    embeddding_model = tool.get_embedding_model()
    loglog.debug(f"embedding模型初始化完成, model_name = {embeddding_model.model_name}")
    

    # 3.建立向量数据库 并存储文本向量
    faiss_db = _create_db(documents, embeddding_model)
    loglog.debug(f"向量数据库创建完成，并存储文本向量, db_name = {EMBEDDING_DB_NAME}")

    # 4.返回向量数据库对象
    return faiss_db