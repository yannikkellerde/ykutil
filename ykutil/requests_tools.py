import requests
import time


def wait_for_url(url: str, max_tries: int = 60, timeout: int = 2) -> bool:
    for _ in range(max_tries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False
