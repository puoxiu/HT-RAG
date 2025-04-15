import torch
from logger import MyLogger

#=============================================== 数据集的配置 ==========================================#

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"

SOURCE_URL_FILE = "../data/source_urls.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 1000  # 分片大小
CHUNK_OVERLAP = 100  # 分片重叠大小

#=============================================== 嵌入 re-rank 向量数据库 的配置 ==========================================#

EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DB_NAME = "faiss_db_{CHUNK_SIZE}_{CHUNK_OVERLAP}.index"   # 向量数据库名称, 存储路径

RETRIEVER_TOP_K = 3  # 检索器返回的文档数量

RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

#================================================= LLM的配置 =============================================#
# 如果使用openai的模型
BASE_URL_OPENAI = "https://www.dmxapi.cn/v1"
LLM_MODEL_OPENAI = "gpt-4o-mini"
API_KEY_OPENAI = "sk-5geduntoNvm5OTAla9PDcnNnzErVD3zFSp8ZM0OxvrCPyT5ox"
# 如果使用glm模型
LLM_MODEL_GLM = "glm-plus"

# 如果使用ollama本地模型
LLM_MODEL_OLLAMA = "qwen-7b"

# 实际使用的模型
LLM_MODEL = LLM_MODEL_OPENAI



loglog = MyLogger().get_logger()
