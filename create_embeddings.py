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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import manifold

from models import SVAL
from configs import get_cfg_defaults
from datasets import PolyvoreDataset
from utils import create_logger

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
DPI = 100

PERPLEXITY = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LR = "auto"
STEPS = 5000

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

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
    parser.add_argument(
        "--device-id", dest="device_id", type=int, default=0,
        help="GPU device id to be used, CPU yet not supported"
    )
    parser.add_argument(
        "--phase", dest="phase", type=str, default="train",
        help="Whether to use train or val images for viz"
    )
    parser.add_argument(
        "--tsne", action='store_true',
        help="Provide this flag to do tsne manually"
    )

    return parser.parse_args()

def transform_image(image, new_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    image = np.transpose(image, (1, 2, 0))  #CHW->HWC
    image = cv2.resize(image, new_size)[:, :, ::-1] #BGR->RGB

    return torch.from_numpy(image.copy()).permute(2, 0, 1) #HWC->CHW

def do_tsne(features, images, save_dir):
    # scale_factor_x = images[0].shape[1]
    # scale_factor_y = images[0].shape[0]
    scale_factor = np.array(
        [[64, 64]], dtype=np.float32
    )
    print("Scale factor: ", scale_factor)
    features = torch.stack(features, dim=0).numpy()

    for perplextiy in PERPLEXITY:
        print("perplexity: ", perplextiy)
        tsne = manifold.TSNE(
            n_components=2,
            init='random',
            random_state=0,
            perplexity=perplextiy,
            learning_rate=LR,
            n_iter=STEPS
        )

        print("Running tsne...")
        Y = tsne.fit_transform(features)

        # Translating space to positive values
        Y = Y + (-1*Y.min(axis=0)[np.newaxis, :]) + 1
        # Scalling the space for better viz
        Y = Y * scale_factor
        # Inting it for coordinates
        Y = Y.astype(np.int32)

        max_w, max_h = Y.max(axis=0)
        max_w += IMAGE_WIDTH
        max_h += IMAGE_HEIGHT

        print("Running rendering...")
        fig = plt.figure(figsize=(max_w/DPI, max_h/DPI), dpi=DPI)

        for coord, image in tqdm(zip(Y, images)):
            fig.figimage(image.permute(1, 2, 0).numpy(), coord[0], coord[1])

        save_path = os.path.join(save_dir, f"tsne_size-{Y.shape[0]}_perpx-{perplextiy}_lr-{LR}")
        fig.savefig(save_path, dpi=DPI)

def main():
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.general.is_training = False
    cfg.save_load.load_models = True
    cfg.save_load.pretrained = True
    cfg.save_load.load_path = args.ckpt_model
    cfg.data.apply_norm = False
    cfg.general.device_id = int(args.device_id)
    cfg.freeze()
    logger = create_logger(
        os.path.join(os.path.dirname(args.ckpt_model), args.phase + "_emb_logs.txt")
    )

    print("Creating dataset")
    dataset = PolyvoreDataset(cfg, logger, phase=args.phase)
    no_images = len(dataset)
    print("Creating model")
    model = SVAL(cfg, logger, None, no_images)

    feature_mats = []
    image_mats = []
    label_list = []

    count = args.image_count if args.image_count > 0 else no_images
    if args.phase == 'train':
        sample_indices = np.random.randint(low=0, high=94000, size=count).tolist()
    else:
        sample_indices = list(range(count))

    with torch.no_grad():
        for i in tqdm(sample_indices):
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


        if args.tsne:
            msg = "Doing tsne manually"
            print(msg)
            logger.info(msg)
            do_tsne(feature_mats, image_mats, args.save_dir)
        else:
            print("Creating summary writer")
            writer = SummaryWriter(args.save_dir)
            msg = "Writing embeddings to tfb file"
            print(msg)
            logger.info(msg)
            writer.add_embedding(
                torch.stack(feature_mats, dim=0),
                metadata=label_list,
                label_img=torch.stack(image_mats, dim=0),
                global_step=0
            )

            writer.close()

if __name__ == '__main__':
    main()
