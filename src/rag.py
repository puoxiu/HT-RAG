from langchain_community.vectorstores import FAISS
import urls
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_community.document_transformers import LongContextReorder
from langchain.chains import LLMChain

import jieba
import torch

from tools import tool
from config import EMBEDDING_DB_NAME, RETRIEVER_TOP_K, RERANKER_MODEL_NAME, DEVICE, loglog
# from llm.llm_openai import get_ChatOpenAI
from llm.chatglm import get_LLM

class RAG():
    def __init__(self, use_ensemble: bool):
        """
        :param use_ensemble: 是否使用混合分词/检索
        """
        self.retriver = self._get_embedding_retriever(use_ensemble)
        self.llm = get_LLM()  # LLM模型
        # reranker
        
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
        self.model.eval()


    # 重写问题
    def _rewrite_question(self, question):
        prompt_template = """
            现在我需要向您咨询有关股票的问题。请按照以下指导原则改写我的问题，以确保它适合用于数据库查询或搜索引擎检索：
            - 将原始问题转换成明确、简洁且针对性强的形式；
            - 确保问题表述专业，避免使用口语化或非正式的语言；
            - 维持问题的原始含义，不得添加或删除任何重要信息；
            - 如果原始问题是开放式的，请尝试将其转化为具体的、可搜索的形式；
            - 如果可能，考虑加入关键术语或技术词汇来增强查询的精确度。

            请直接提供改写后的问题，注意不要改变问题的语义，不要引入新的意思。
            下面是原始的问题：
            {question}
            ---
            请直接给出改写后的问题，不要添加任何额外的解释或信息。
            """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        rewrite_chain = prompt | self.llm
        rewrite = rewrite_chain.invoke({"question": question}).content

        print(f"[rewrite: ] \n\tinit is: {question} \n\tnow  is: {rewrite}\n")
        return rewrite

    # 获取检索器
    def _get_embedding_retriever(self, use_ensemble: bool):
        """
        use_ensemble: 是否使用混合分词
        """

        embedding_model = tool.get_embedding_model()

        # 用于重排检索结果的实例
        # reranker = _get_reranker()

        # 用于存储和检索文档的嵌入向量
        db = FAISS.load_local(
            f"data/{EMBEDDING_DB_NAME}",
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

        # 如果不需要混合分词
        if not use_ensemble:
            return retriever

        # 重新生成docs耗时过久, 使用缓存
        docs = tool.get_contents(tuple(urls.valid_urls))
        loglog.debug(f"建立bm25检索器, 当前文档数量: len(docs) = {len(docs)}")

        bm25_retriever = BM25Retriever.from_documents(
            docs,
            k=RETRIEVER_TOP_K,
            bm25_params={"k1": 1.5, "b": 0.75},
            preprocess_func=jieba.lcut  # 中文需要使用jieba分词
        )
        ensemble_retriever = EnsembleRetriever(
                                retrievers=[bm25_retriever, retriever],
                                weights=[0.4, 0.6])

        return ensemble_retriever


    # 根据问题获取相关文档
    def get_context(self, question: str,is_rewrite: bool, score_threshold = 0):
        """
        根据问题获取相关文档
        :param question: 问题
        :param score_threshold: 分数阈值
        :param is_rewrite: 是否重写问题
        :return: 文档
        """
        if is_rewrite:
            re_question = self._rewrite_question(question)

        # 余弦相似度召回
        docs = self.retriver.invoke(question)
        re_docs = self.retriver.invoke(re_question) if is_rewrite else None
        
        # 得到文档的分数
        def _get_docs_scores(docs):
            pairs = [[question, doc.page_content] for doc in docs]
            with torch.no_grad():
                # 使用重排模型进行评分
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            # 确保scores是一维的
            scores = scores.squeeze()
            return scores

        scores = _get_docs_scores(docs)
        doc_scores = list(zip(docs, scores.tolist()))
        if is_rewrite:
            re_scores = _get_docs_scores(re_docs)
            re_doc_scores = list(zip(re_docs, re_scores.tolist()))
            # doc_scores.extend(re_doc_scores)
            doc_scores = doc_scores + re_doc_scores

        # 从大到小排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        # 过滤掉分数低于阈值的文档
        loglog.debug(f"当前检索到的文档数量: {len(doc_scores)}")
        sorted_docs = [doc for doc, score in doc_scores if score >= score_threshold][:RETRIEVER_TOP_K]

        # 长上下文检索器
        reordered_docs = LongContextReorder()
        reordered_docs = reordered_docs.transform_documents(sorted_docs)

        return "\n\n".join(doc.page_content for doc in reordered_docs)  


    # 根据问题获取答案
    def get_answer(self, question: str,is_rewrite: bool, verbose = False):
        contex = self.get_context(question, is_rewrite)
        
        PROMPT_TEMPLATE = """
        作为一个专门用于问答的人工智能助手，我会使用给定的上下文来为你的查询提供清晰简洁的回答，回答不超过五句话，并且不会添加任何诸如“好的”或“从资料中可以得出”这样的大语言模型常用的填充词。如果没有可用的必要信息，我会如实告知你。
        回答要简洁。如果问题有明确答案，则只需回答答案；若没有相关答案，你直接回复： '对不起，暂时不能回答这个问题。'。
        上下文：
        {context}
        ---
        问题：
        {question}
        ---
        尽量用中文回答，术语和专有名词除外。
        注意：对于格式为“input_field”: 问题, “output_field”: “答案” 的问题，请仅提供答案值，无需额外文本。
        以下是你的答案：
        """

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # todo: momory模块，是否需要有历史记忆？？？
        # chain = LLMChain(
        #     llm=self.llm,
        #     prompt=prompt,
        #     verbose=verbose,
        #     memory=None,
        # )

        chain = prompt | self.llm

        # res = chain.predict(
        #     context=contex,
        #     question=question
        # )
        res = chain.invoke(
            {"context": contex, "question": question}
        ).content

        return res
