"""ex02_14.py"""
def wget_resource(url, out, filename):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    r = session.get(url)
    
    out(r.content, filename)

def stream_out(content, *args):
    print(content)

def file_out(content, filename):
    with open(filename, 'wb') as f:
        f.write(content)

def main():
    url = "http://google.com/favicon.ico"
    filename = url.split('/')[-1]

    wget_resource(url, file_out, filename)

if __name__ == '__main__':
    main()