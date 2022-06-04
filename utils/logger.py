import os
import sys
import logging

def create_logger(logfile_path):
    ft = "%(asctime)s %(module)s : [%(levelname)s #%(lineno)d]: %(message)s"
    logging.basicConfig(
        level = logging.INFO,
        format = ft,
        handlers = [logging.FileHandler(logfile_path),
                    #logging.StreamHandler(sys.stdout)
        ],
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger("root")
    return logger