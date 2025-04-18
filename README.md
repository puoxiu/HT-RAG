# HT-RAG 📚

## 一、系统简介 🤖
本RAG（Retrieval-Augmented Generation）系统基于知识库搭建，借助了 `langchain` 框架与 `faiss` 向量数据库。嵌入模型选用 `bge - m3`，重排序模型采用 `BAAI/bge - reranker - large`，问答模型则为 `chatglm`。

## 二、运行 🚀

### 2.1 环境搭建
```sh
git clone https://github.com/puoxiu/HT-RAG.git
cd HT-RAG
pip install -r requrirements.txt
```

### 2.2 注意事项
本系统采用 `langchain` 的 `WebBaseLoader` 作为加载器，因此需要依据 `config` 配置准备好知识库的 `urls`。

## 三、评估方式 📊
- **召回阶段**：评估召回率、命中率；
- **回答阶段**：基于验证集，计算每个回答与验证集对应答案的余弦相似度，并求出平均值

## 四、改进方案 💪

### 4.1 初始评估
仅运用 `bge - m3` 作为嵌入模型、`chatglm` 作为大语言模型（LLM），对验证集开展问答评估，所有数据的平均余弦相似度为 **0.79**；

### 4.2 嵌入模型微调
发现 `bge - m3` 嵌入模型存在缺陷，即不相似的文本计算出的相似度得分偏高。对嵌入模型进行微调后，召回率提升了 **4%**，但 RAG 的相似度得分降至 **0.76**；

### 4.3 进一步改进
采取了一系列改进措施，包含使用 `bm25` 和向量的混合检索、对输入问题进行重写（rewrite）、引入重排序（rerank）模型对召回结果进行重新排序等，最终将平均相似度提高至 **0.87**

### 4.4 最终模型的消融评估
| 是否使用 rewrite | 是否使用混合精度检索 | 是否重排序 | 平均余弦相似度 |
| :---: | :---: | :---: | :---: |
| 是 | 是 | 是 | **[0.87]** |
| 是 | 是 | 否 | **[0.81]** |
| 是 | 否 | 是 | **[0.80]** |
| 否 | 是 | 是 | **[0.87]** |
| 否 | 否 | 否 | **[0.76]** |

`rewrite` 的作用有点「薛定谔」，不管了很多时候“感觉”还算是有用的

## 五、运行模式 🎯

### 5.1 问答模式
```sh
python -init 1 -mode 0
```

### 5.2 评估模式
```sh
python -init 1 -mode 1
```

## 六、一些小细节 🧐

### 6.1 bge 模型微调
#### 6.1.1 微调方式
微调方式更新迅速，建议每次都参考官网获取最新信息

#### 6.1.2 数据集
利用大语言模型（LLM）构造出 **200** 条三元组：[问题，正例]。负例构造方式如下：
使用基础模型召回文本 `top20`，对于所有的 `query`：
- 约 **30%** 选取距离 `dis < 0.5` 的召回文本作为负例；
- 约 **70%** 选取距离 `0.5 < dis < 0.7` 的召回文本作为负例；
- 部分负例添加了「对不起，我不知道；不知道」等内容。

### 6.2 文档内容加载缓慢
由于 `url` 数量较多，约 **200** 个左右，在运行前需进行访问 `access` 校验，耗时较长。解决办法是使用多线程（此处暂未明确是协程还是多线程，后续需深入了解 Python 并发的实际实现方式，当前借助大模型生成的代码已解决该问题）；

### 6.3 显存爆炸
实验中 `url` 数量达到 **200** 个，分割后的 `documents` 数据量庞大，占用大量内存和 `CPU`。解决方案是效仿训练过程，设置批处理，每次向 `faiss` 中加载 `batch_size` 个 `documents`；

### 6.4 embedding 模型相似度问题
使用 `[BAAI/bge - large - zh - v1.5]` 模型时效果不佳。例如，对于两个句子：
- sentence1 = "是的，后面我们会加入 rpm 中，目前编译部署后可以使用 lgraph_peer"
- sentence2 = "对不起，我不太明白你的意思。"
计算它们的余弦相似度为 **0.6209**。
解决方法是对 `embedding` 模型进行微调