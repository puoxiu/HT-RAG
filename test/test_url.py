import requests



def _validate_urls(url: str) -> bool:
    """
    校验url的合法性
    """
    session = requests.Session()
    # session.headers.update({"User-Agent": USER_AGENT})
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        return True
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
     

if __name__ == "__main__":
    url = "https://wdndev.github.io/llm_interview_note/#/09.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/1.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B9%BB%E8%A7%89/1.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B9%BB%E8%A7%89"
    if _validate_urls(url):
        print(f"{url} is valid.")
