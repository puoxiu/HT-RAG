# 工具类
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from config import EMBEDDING_MODEL_NAME, DEVICE, CHUNK_SIZE, CHUNK_OVERLAP
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import functools



def get_embedding_model():
    """
    获取embedding模型
    """
    model_name = EMBEDDING_MODEL_NAME
    model_kwargs = {"device": DEVICE}
    encode_kwargs = {"normalize_embeddings": True}
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    return embedding


@functools.lru_cache(maxsize=1)
# def get_contents(urls: list[str]):
def get_contents(urls: tuple[str]):
    """
    获取url网页内容
    """
    loader = WebBaseLoader(urls)
    documents = loader.load()

    separators = ["\n\n", "\n", " ", ".", "。", ";", "；",]  # 定义分割符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, 
        separators=separators, 
        keep_separator=False,  # 是否保留这些分割符在文本块中
    )
    documents = text_splitter.split_documents(documents)
    return documents


def clear_cache():
    get_contents.cache_clear()