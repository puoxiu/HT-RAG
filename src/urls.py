from config import SOURCE_URL_FILE, USER_AGENT
import requests

valid_urls: list[str] = []  # 用于存储有效的URL


def _add_urls():
    new_urls = []
    """
    获取新url的方法， 目前暂时不需要
    """

    return new_urls


def _validate_urls(url: str) -> bool:
    """
    校验url的合法性
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    try:
        response = session.get(url)
        response.raise_for_status()  # 检查请求是否成功
        return True
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return False


def init_urls():
    """
    在这里可以添加新的url：使用爬虫、读取文档等方式
    """
    global valid_urls
    
    with open(SOURCE_URL_FILE, "r") as f:
        urls = [line.strip() for line in f.readlines()]

    urls.extend(_add_urls())
    urls = list(set(urls))  # 去重

    # 得到所有有效访问的url
    valid_urls = [url for url in urls if _validate_urls(url)]

    print(f"当前所有有效url如下:\n")
    print("\n".join(valid_urls))
    print(f"当前有效url数量：{len(valid_urls)}")

