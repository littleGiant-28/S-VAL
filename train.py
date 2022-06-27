import os
import random
import argparse
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from models import SVAL
from utils import create_logger, TFBLogger, get_timestamp
from configs import get_cfg_defaults
from datasets import get_polyvore_dataloader

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SVAL with provided config file"
    )
    parser.add_argument("--config", "-cfg", type=str, required=False,
                        help="Path to the YAML config file")
    return parser.parse_args()

def perform_prerequistes(args):
    #Prepare config
    cfg = get_cfg_defaults()
    if args.config:
        if not os.path.exists(args.config):
            msg = "Provided config file does not exists: {}".format(args.config)
            logger.info(msg)
            raise FileNotFoundError(msg)
        print("Using config file at {}".format(args.config))
        cfg.merge_from_file(args.config)

    #Prepare dirs
    if not os.path.isdir(cfg.save_load.exp_root):
        print("The experiment root dir '{}' does not exist, creating it"
              .format(cfg.save_load.exp_root))
        os.mkdir(cfg.save_load.exp_root)

    timestamp = get_timestamp()
    exp_name = cfg.save_load.exp_name + "_" + timestamp
    exp_path = os.path.join(cfg.save_load.exp_root, exp_name)
    if os.path.exists(exp_path):
        msg = """Critical error, the experiment path \
                already exists: {}""".format(exp_path)
        raise Exception(msg)
    os.mkdir(exp_path)
    
    cfg.save_load.exp_path = exp_path
    tfb_log_dir = os.path.join(exp_path, cfg.logging.tfb_log_dir)
    cfg.logging.tfb_log_dir = tfb_log_dir
    os.mkdir(tfb_log_dir)
    cfg.freeze()

    #Create final config yaml in experiment dir
    with open(os.path.join(exp_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    #Init logging
    log_path = os.path.join(exp_path, cfg.logging.log_file)
    logger = create_logger(log_path)
    logger.info("Experiment name: {}".format(exp_name))
    logger.info("Experiment path: {}".format(exp_path))
    if args.config:
        logger.info("Config file used: {}".format(args.config))
    logger.info("Copied used config data as config.yaml to experiment dir")
    logger.info("-------------------------------------------------------------")
    logger.info("The command line parameters are as follows:\n{}"
                .format(cfg.dump()))
    logger.info("-------------------------------------------------------------")
    logger.info("Creating Tensorboard logger")
    tfb_logger = TFBLogger(cfg)

    return cfg, logger, tfb_logger

def update_avg_states(avg_stats, new_stats, step):
    for key, value in new_stats.items():
        avg_stats[key] = ((avg_stats[key]*step) + value) / (step+1)

def train_loop(config, model, dataloaders, all_loggers):
    start_epoch = model.start_epoch
    total_steps = model.total_steps
    total_epochs = config.train.epochs

    train_dataloader, test_dataloader = dataloaders
    logger, tfb_logger = all_loggers

    if not (config.save_load.pretrained or config.save_load.load_models):
        model.initalize_memory(train_dataloader)
    
    for current_epoch in range(start_epoch, total_epochs):
        tepoch = tqdm(range(0, len(train_dataloader)), unit="batch")
        last_step = len(train_dataloader)-1
        tepoch.set_description("Epoch: {}".format(current_epoch))
        dataloader_iter = iter(train_dataloader)
        
        for current_step in tepoch:
            batch = dataloader_iter.next()
            is_last_step = (current_step==last_step)
            stats = model.step(
                batch, current_epoch, current_step, is_last_step, total_steps
            )
            
            tepoch.set_postfix(**stats)

            total_steps += 1
            
        model.save_model(current_epoch, total_steps)
        model.memory_index_reset()

        if config.train.do_val:
            tepoch = tqdm(range(0, len(test_dataloader)), unit="batch")
            tepoch.set_description("Test Epoch: {}".format(current_epoch))
            dataloader_iter = iter(test_dataloader)

            avg_stats = defaultdict(int)

            for current_step in tepoch:
                batch = dataloader_iter.next()
                stats = model.test_step(batch)

                tepoch.set_postfix(**stats)
                update_avg_states(avg_stats, stats, current_step)

                total_steps += 1

            logger.info("Validation set wise average results: ")
            logger.info(
                "Epoch: {} Hist loss: {} Hist local loss: {} Hist global loss: {} "
                "SLPD Loss: {} TD Loss: {} Total Loss: {}"
                .format(current_epoch, avg_stats['hist_loss'], 
                avg_stats['hist_loss_local'], avg_stats['hist_loss_global'], 
                avg_stats['slpd_loss'], avg_stats['td_loss'],
                avg_stats['total_loss'])
            )

            tfb_logger.log_scalars(
                current_epoch, list(avg_stats.keys()), list(avg_stats.values()),
                prefix="test/"
            )

def main():
    args = parse_args()
    cfg, logger, tfb_logger = perform_prerequistes(args)
    
    # torch.manual_seed(cfg.general.seed)
    # random.seed(cfg.general.seed)
    # np.random.seed(cfg.general.seed)
    
    logger.info("Constructing training dataloader")
    print("Constructing training dataloader")
    train_dataloader, test_dataloader, no_image = get_polyvore_dataloader(cfg, logger)
    test_dataloader = None  #deleting test_dataloader
    dataloaders = (train_dataloader, test_dataloader)
    
    print("Constructing SVAL class instance")
    logger.info("Constructing SVAL class instance")
    model = SVAL(cfg, logger, tfb_logger, no_image)
    logger.info("SVAL class instance intialized sucessfully")
    
    train_loop(cfg, model, dataloaders, (logger, tfb_logger))

if __name__ == '__main__':
    main()

