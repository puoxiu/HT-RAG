from embedding import init_embeddings
from rag_call import answer
import urls





if __name__ == '__main__':
    urls.init_urls()

    if len(urls.valid_urls) == 0:
        print("没有有效的URL可供处理, 请检查配置文件或添加新的URL!")
        exit(1)

    init_embeddings(urls.valid_urls)

    print("嵌入模型初始化完成。")
    print("请开始提问：")

    
    while True:
        question = input("请输入问题(输入exit退出):")
        if question.lower() == "exit":
            break
        # question = "请问 TuGraph 的存储过程？"
        answer_result = answer(question)
        print(answer_result)

    