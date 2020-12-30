import requests
import time
from urllib.robotparser import RobotFileParser
from requests.compat import urlparse, urljoin
from requests.exceptions import HTTPError

def canfetch(url, agent='*', path='/'):
    robot = RobotFileParser(urljoin(url, '/robots.txt'))
    robot.read()
    return robot.can_fetch(agent, urlparse(url)[2])


def download(url, params={}, headers={}, method='GET', limit=3):
    method = method.upper()
    if canfetch(url) == False:
        print('[Error] ' + url)
    else:
        try:
            resp = requests.request(method, url,
                                    params=params if method == 'GET' else {},
                                    data=params if method == 'POST' else {},
                                    headers=headers)
            resp.raise_for_status()
        except HTTPError as e:
            if limit > 0 and e.response.status_code >= 500:  # 500 에러 재귀 ;너무많은 트래픽 안생기도록
                print(limit)
                time.sleep(300)
                resp = download(url, params, headers, method, limit - 1)
            else:
                print('[{}] '.format(e.response.status_code) + url)
                print(e.response.status_code)
                print(e.response.reason)
                print(e.response.headers)
        return resp