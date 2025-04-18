from config import SOURCE_URL_FILE, USER_AGENT
import requests
import concurrent.futures


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
        response = session.get(url, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        return True
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return False




def init_urls(is_init: bool):
    global valid_urls

    with open(SOURCE_URL_FILE, "r") as f:
        urls = [line.strip() for line in f.readlines()]

    urls.extend(_add_urls())
    urls = list(set(urls))  # 去重

    if is_init:
        def validate_single_url(url):
            return url if _validate_urls(url) else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(validate_single_url, urls))
        valid_urls = [url for url in results if url is not None]

        print(f"当前所有有效url如下:\n")
        print("\n".join(valid_urls))
        print(f"当前有效url数量：{len(valid_urls)}")

        with open(SOURCE_URL_FILE, "w") as f:
            for url in valid_urls:
                f.write(url + "\n")
    else:
        valid_urls = urls

# def init_urls(is_init: bool):
#     """
#     在这里可以添加新的url：使用爬虫、读取文档等方式
#     """
#     global valid_urls

#     with open(SOURCE_URL_FILE, "r") as f:
#         urls = [line.strip() for line in f.readlines()]

#     urls.extend(_add_urls())
#     urls = list(set(urls))  # 去重

#     # 得到所有有效访问的url
#     if is_init:
#         valid_urls = [url for url in urls if _validate_urls(url)]
#         print(f"当前所有有效url如下:\n")
#         print("\n".join(valid_urls))
#         print(f"当前有效url数量：{len(valid_urls)}")

#         # 将有效的 URL 重新写回到文件中
#         with open(SOURCE_URL_FILE, "w") as f:
#             for url in valid_urls:
#                 f.write(url + "\n")
#     else:
#         valid_urls = urls

if __name__ == "__main__":
    import os
    print(f"当前核心数量：{os.cpu_count()}")
    print(f"当前线程数量：{min(os.cpu_count() + 4, 32)}")
    init_urls(True)
    print(f"当前所有有效url如下:\n")
    print("\n".join(valid_urls))
    print(f"当前有效url数量：{len(valid_urls)}")