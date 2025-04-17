from embedding import init_embeddings
from rag_call import answer
from evaluation import evaluate
import urls
import argparse

# https://abdullin.com/ilya/how-to-build-best-rag/


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--init', type=int, help='是否初始化向量数据库: 0-不初始化, 1-初始化', required=True)
parser.add_argument('-m', '--mode', type=int, default=0, help='运行模式: 0-对话模式, 1-评估模式')



if __name__ == '__main__':
    args = parser.parse_args()

    urls.init_urls()
    if len(urls.valid_urls) == 0:
        print("没有有效的URL可供处理, 请检查配置文件或添加新的URL!")
        exit(1)
    
    if args.init == 1:
        init_embeddings(urls.valid_urls)

    
    if args.mode == 0:
        print("请开始提问：")
        while True:
            question = input("请输入问题(输入exit退出):")
            if question.lower() == "exit":
                break
            # question = "请问 TuGraph 的存储过程？"
            answer_result = answer(question)
            print(answer_result)
    elif args.mode == 1:
        evaluate()

    