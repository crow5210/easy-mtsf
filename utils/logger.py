from torch.utils.tensorboard import SummaryWriter
import logging
import os,time,logging

def init_tensorboard(ckpt_save_dir):
    tensorboard_writer = SummaryWriter(os.path.join(ckpt_save_dir, 'tensorboard'))
    return tensorboard_writer

def init_logger(log_name,ckpt_save_dir):
    logger = logging.getLogger(log_name)
    if len(logger.handlers) == 0:
        log_level = logging.INFO
        log_file_name = '{}_{}.log'.format("training_log", time.strftime('%Y%m%d%H%M%S', time.localtime()))
        log_file_path = os.path.join(ckpt_save_dir, log_file_name)
        logger.propagate = False
        logger_handlers = [logging.StreamHandler(),logging.FileHandler(log_file_path, 'w')]
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in logger_handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)
        logger.setLevel(log_level)
    return logger