# Modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
import logging
import os
import sys
import time
import datetime
import tensorflow as tf

from anomaly.apis.utils import color

def setup_logger(name, save_dir, distributed_rank, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    # formatter = logging.Formatter("[%(asctime)s] %(name)s [%(levelname)s]: %(message)s",
    formatter = logging.Formatter("{now} - {name} - {level} - {message}".format(
            now = '%(asctime)s',
            name = '%(name)s',
            level = '%(levelname)s',
            message = '%(message)s'
        ), "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if filename is None:
            filename = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()) + ".log"
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_tblogger(save_dir, distributed_rank):
    if distributed_rank>0:
        return None
    from tensorboardX import SummaryWriter
    tbdir = os.path.join(save_dir,'tb')
    os.makedirs(tbdir,exist_ok=True)
    tblogger = SummaryWriter(tbdir)
    return tblogger


#added
class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        with tf.compat.v1.Graph().as_default():
            self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
###