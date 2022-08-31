"""ex02_15.py"""
from threading import Thread

class SimpleThread(Thread):
    def __init__(self):
        super(SimpleThread, self).__init__()
        self._callback = None
        self.args = None

    def callback(self, func, *args, **kwargs):
        self._callback = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self._callback:
            self._callback(*self.args, **self.kwargs)

class SimpleWGet:
    def __init__(self):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get(self, url):
        r = self.session.get(url)
        filename = url.split('/')[-1]

        with open(filename, 'wb') as f:
            f.write(r.content)

def main(): 
    OPENCV_SIMPLE_DATA = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"
    urls = [
        OPENCV_SIMPLE_DATA + "ml.png",
        OPENCV_SIMPLE_DATA + "home.jpg",
        OPENCV_SIMPLE_DATA + "butterfly.jpg",
        OPENCV_SIMPLE_DATA + "baboon.jpg",
        OPENCV_SIMPLE_DATA + "lena.jpg",
        OPENCV_SIMPLE_DATA + "lena.jpg"
    ]

    wget = SimpleWGet()

    for url in urls:
        th = SimpleThread()
        th.callback(wget.get, url)
        th.start()

    th.join()

if __name__ == '__main__':
    main()
