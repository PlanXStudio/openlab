"""ex02_13.py"""
from threading import Thread
from stream_logger import StreamLogger
from random import randint

def work_thread(d, result, pos, logger):
    logger.info(d)
    result[pos] = sum(d)

def main():
    logger = StreamLogger()
    data = [randint(1, 10) for i in range(100)]

    result = [0] * 10
    logger.info(f"{result = }")

    for i in range(10):
        th = Thread(target=work_thread, args=(data[i*10:i*10+10], result, i, logger))
        th.start()

    th.join()
    total = sum(result)
    logger.info(f"{result = }, {total = }")

if __name__ == '__main__':
    main()
