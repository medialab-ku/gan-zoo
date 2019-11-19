import logging
import os


def init_logger(name, path, level=20):
    # log_dir = 'ckpt/{comment}'.format(comment=name)
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = os.path.join(path, 'debug.log')

    form = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    file = logging.FileHandler(log_file)
    stream = logging.StreamHandler()

    file.setFormatter(form)
    stream.setFormatter(form)

    logger = logging.getLogger()  # only use root logger
    logger.setLevel(level)  # CRIT=50, ERR=40, WARN=30, INFO=20, DEBUG=10
    logger.addHandler(file)
    logger.addHandler(stream)

    return logger
