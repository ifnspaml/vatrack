import logging

from mmcv.utils import get_logger


def get_root_logger(model_name=None, log_file=None, log_level=logging.INFO):
    return get_logger(name=model_name, log_file=log_file, log_level=log_level)
