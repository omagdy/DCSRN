import logging

logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.DEBUG)

def log(log):
	print(log)
	logging.info(log)
