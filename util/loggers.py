import logging
import os
from tensorboardX import SummaryWriter



def get_logger(log_dir, experiment_name):
    os.makedirs(log_dir, exist_ok=True) 
    log_filename = os.path.join(log_dir,experiment_name+".log")
    file_handler = logging.FileHandler(log_filename)

    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    return logger

def get_tb_logger(tb_dir,experiment_name):
    tb_experiment_dir = os.path.join(tb_dir,experiment_name)
    os.makedirs(tb_dir, exist_ok=True) 
    os.makedirs(tb_expe