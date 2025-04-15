
import sys, os
from loguru import logger

# 获得当前项目的绝对路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(root_dir, "./logs")

if not os.path.exists(log_dir): 
    os.mkdir(log_dir)

LOG_FILE = "rag.log"  # 存储日志的文件

# Trace < Debug < Info < Success < Warning < Error < Critical

class MyLogger:
    def __init__(self):
        log_file_path = os.path.join(log_dir, LOG_FILE)
        # 写日志的对象
        self.logger = logger 
        # 清空所有设置
        self.logger.remove()
        # 添加控制台输出的格式,sys.stdout为输出到屏幕
        # level='DEBUG'表示输出的最低等级, 低于则不输出
        self.logger.add(sys.stdout, level='DEBUG',
                        format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 线程名
                               "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                               ":<cyan>{line}</cyan> | "  # 行号
                               "<level>{level}</level>: "  # 等级
                               "<level>{message}</level>",  # 日志内容
                        )
        # 输出到文件的格式,注释下面的add,则关闭日志写入
        self.logger.add(log_file_path, level='DEBUG', encoding='UTF-8',
                        format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 进程名
                               '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
                        rotation="10 MB",  # 日志文件生成的规则  rotation="1 week"  rotation="1 days"
                        retention=20  # 保留日志文件的规则
                        )

    def get_logger(self):
        return self.logger




if __name__ == '__main__':
    loglog = MyLogger().get_logger()
    
    loglog.debug("This is a debug message.")
    loglog.info("This is an info message.")
    loglog.warning('这是一个警告')
    loglog.trace('xxxx')
    loglog.error('这是一个错误')
    # print('str.pdf'['str.pdf'.rindex('.'):])
    # # @log.catch  # 整个函数自动加上try， catch。自动捕获异常，并且通过日志打印
    # def test():
    #     try:
    #         print(3/0)
    #     except ZeroDivisionError as e:
    #         log.error(e)
    #         # log.exception(e)  # 打印异常信息
    #         log.exception(e)
    # test()
