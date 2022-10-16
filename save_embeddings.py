import os
import argparse

import yaml
import torch
from tqdm import tqdm

from models import SVAL
from utils import create_logger
from configs import get_cfg_defaults
from datasets import PolyvoreDataset

EXP_CONFIG_FILE_NAME = "config.yaml"
SAVE_DEVICE = torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save embeddings for outfit gen training"
    )
    parser.add_argument(
        "--config", type=str, required=True, 
        help="Path to the config file to use for saving embeddings"
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", type=str, required=True,
        help="Path to the dir to save embeddings, will be created if not exist"
    )
    return parser.parse_args()

def find_model_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    weight_dir = os.path.dirname(config["save_load"]["load_path"])
    config_path = os.path.join(weight_dir, EXP_CONFIG_FILE_NAME)

    return config_path

def main():
    args = parse_args()

    cfg = get_cfg_defaults()
    model_config = find_model_config(args.config)
    cfg.merge_from_file(model_config)
    cfg.merge_from_file(args.config)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    logger = create_logger(os.path.join(args.save_dir, "emb_logs.txt"))

    with torch.no_grad():
        print("Creating dataset...")
        dataset = PolyvoreDataset(cfg, logger, phase="test")
        no_images = len(dataset)
        print("Total images: ", no_images)
        print("Creating model...")
        model = SVAL(cfg, logger, None, no_images)
        model.feature_model.eval()

        print("Saving embeddings...")
        for index in tqdm(range(no_images)):
            sample = dataset[index]
            image = sample['image'].unsqueeze(0).to(model.device)
            image_name = sample['image_name']

            features = model.feature_model.get_embeddings(image)
            tensor_save_path = os.path.join(args.save_dir, str(image_name) + ".pt")

            torch.save(features.cpu().squeeze(0), tensor_save_path)

if __name__ == '__main__':
    main()