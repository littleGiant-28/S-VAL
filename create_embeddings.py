import os
import pickle
import argparse

#Hack for making tensorboard writing work
#found here:https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import PolyvoreDataset

from models import SVAL
from configs import get_cfg_defaults
from utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for viz in tensorboard"
    )
    parser.add_argument(
        "--ckpt-model", dest="ckpt_model", type=str, required=True,
        help="Path to the ckpt model .pt file"
    )
    parser.add_argument(
        "--config", dest="config", type=str, required=True,
        help="Path to the config file loading models and images"
    )
    parser.add_argument(
        "--save-dir", dest="save_dir", type=str, required=True,
        help="Path to the dir to save embedding data for viz"
    )
    parser.add_argument(
        "--image-count", dest="image_count", type=int, default=1000,
        help="Number of images to project in viz"
    )

    return parser.parse_args()

def transform_image(image, new_size=(64, 64)):
    image = np.transpose(image, (1, 2, 0))  #CHW->HWC
    image = cv2.resize(image, new_size)[:, :, ::-1] #BGR->RGB

    return torch.from_numpy(image.copy()).permute(2, 0, 1) #HWC->CHW

def main():
    args = parse_args()
    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.general.is_training = False
    cfg.save_load.load_models = True
    cfg.save_load.pretrained = True
    cfg.save_load.load_path = args.ckpt_model
    cfg.data.apply_norm = False
    cfg.freeze()
    logger = create_logger(
        os.path.join(os.path.dirname(args.ckpt_model), "emb_logs.txt")
    )

    print("Creating dataset")
    dataset = PolyvoreDataset(cfg, logger, phase="train")
    no_images = len(dataset)
    print("Creating model")
    model = SVAL(cfg, logger, None, no_images)
    print("Creating summary writer")
    writer = SummaryWriter(args.save_dir)

    feature_mats = []
    image_mats = []
    label_list = []

    with torch.no_grad():
        for i in tqdm(range(args.image_count)):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(model.device)
            image_name = sample['image_name']

            features = model.feature_model.get_embeddings(image)

            feature_mats.append(features.squeeze(0).detach().cpu())
            image_mats.append(transform_image(image.squeeze(0).cpu().numpy()))
            label_list.append(image_name)

        del model
        del dataset
        del features
        del image

        writer.add_embedding(
            torch.stack(feature_mats, dim=0),
            metadata=label_list,
            label_img=torch.stack(image_mats, dim=0),
            global_step=0
        )

        writer.close()


if __name__ == '__main__':
    main()