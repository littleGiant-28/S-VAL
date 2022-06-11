import os
import glob
import json
import random

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from utils import get_fg_mask, get_color_histogram

#../data/polyvore_outfits/images/176341004.jpg
class PolyvoreDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, logger):
        super(PolyvoreDataset, self).__init__()
        self.config = config
        self.logger = logger
        self.image_root = self.config.data.image_root
        self.semantic_categories_supported = [
            "tops", "bottoms", "all-body", "outerwear"
        ]
        self.extension = self.config.data.image_extension
        self.hist_bin_size = self.config.model.hist_bin_size
        self.image_resolution = self.config.data.image_resolution
        
        self.image_ids, self.item_categories = self.get_items_list(
            self.config.data.item_json_path
        )
        self.index_item_map = dict(
            zip(list(range(0, len(self.image_ids))), self.image_ids)
        )
        self.item_index_map = dict(
            zip(self.image_ids, list(range(0, len(self.image_ids))))
        )
        self.resnet_normalize = transforms.Normalize(
            mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]
        )
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            self.resnet_normalize
        ])
        
    def get_items_list(self, item_json_path):
        topk = self.config.data.topk
        image_ids = []
        item_categories = []
        with open(item_json_path, 'r') as f:
            item_dict = json.load(f)
            
        for item_id, item_value in item_dict.items():
            if item_value['semantic_category'] in \
                    self.semantic_categories_supported:
                image_ids.append(item_id)
                item_categories.append(item_value['semantic_category'])
        
        self.logger.info("Total images found: {}".format(len(image_ids)))
        
        if topk > 0:
            self.logger.info("Using only first {} images".format(topk))
            image_ids = image_ids[:topk]
            item_categories = item_categories[:topk]
        
        return image_ids, item_categories

    def __len__(self):
        return len(self.image_ids)
    
    def hist_transform(self, hist):
        hist = torch.from_numpy(hist)
        hist = (hist + self.config.data.stability_constant) / hist.sum(dim=1, keepdims=True)
        hist = torch.log(hist)
        
        return hist

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.image_root, str(self.image_ids[idx]) + self.extension
        )
        image = cv2.imread(image_path)
        item_mask = get_fg_mask(image)
        global_color_hist = get_color_histogram(
            image, item_mask, self.hist_bin_size
        )
        global_color_hist = self.hist_transform(global_color_hist)

        sample = {
            'image': self._transform(image),
            'global_color_hist': global_color_hist,
        }
        
        return sample