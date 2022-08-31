"""ex02_11.py"""
import logging

logger = logging.getLogger(__name__)

streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('./mini.log')

formatter = logging.Formatter("[%(asctime)s][%(levelname)-8s|%(filename)s:%(lineno)s] >>> %(message)s")
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

logger.setLevel(level=logging.DEBUG)

logger.debug("it's debug log")
logger.info("it's info log")
logger.warning("it's warning log")
logger.error("it's error log")
logger.critical("it's critical log")