import os
import glob
import json
import pickle
import random

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from utils import get_fg_mask, get_color_histogram, normalize_image

#../data/polyvore_outfits/images/176341004.jpg
class PolyvoreDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, logger, phase):
        super(PolyvoreDataset, self).__init__()
        self.config = config
        self.logger = logger
        self.phase = phase
        self.image_root = self.config.data.image_root
        self.semantic_categories_supported = [
            "tops", "bottoms", "all-body", "outerwear"
        ]
        self.extension = self.config.data.image_extension
        self.hist_bin_size = self.config.model.hist_bin_size
        self.image_resolution = self.config.data.image_resolution
        self.patch_resolution_range = self.config.data.patch_resolution_range
        self.patch_resize_resolution = round(
            self.config.data.patch_resize_resolution * self.image_resolution
        )
        self.patch_sample_mean = np.array([
            self.config.data.patch_sample_mean, 
            self.config.data.patch_sample_mean
        ])
        self.patch_sample_cov = np.array([
            [self.config.data.patch_sample_cov, 0], 
            [0, self.config.data.patch_sample_cov]
        ])
        self.patch_sample_range = self.config.data.patch_sample_range
        self.patch_sample_tries = self.config.data.patch_sample_tries
        self.patch_sample_white_cutoff = self.config.data.patch_sample_white_cutoff
        
        self.split_indices = self.load_split_index()
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

        if self.config.data.apply_norm:
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                self.resnet_normalize
            ])
        else:
            self._transform = transforms.ToTensor()

    def load_split_index(self):
        file_path = self.config.save_load.train_split if self.phase == 'train' \
                        else self.config.save_load.test_split
        if not os.path.exists(file_path):
            raise Exception("Split pickle file {} not found".format(file_path))
        with open(file_path, 'rb') as f:
            indices = pickle.load(f)

        return indices
        
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

        image_ids = [image_ids[ind] for ind in self.split_indices]
        item_categories = [item_categories[ind] for ind in self.split_indices]

        self.logger.info("This split uses images {}".format(len(image_ids)))
        
        if topk > 0:
            self.logger.info("Using only first {} images".format(topk))
            image_ids = image_ids[:topk]
            item_categories = item_categories[:topk]
        
        return image_ids, item_categories

    def __len__(self):
        return len(self.image_ids)

    def get_patch_image(self, image, mask):
        #center = self.image_resolution // 2
        patch_resolution = round(
            self.image_resolution * random.uniform(*self.patch_resolution_range)
        )
        #line equation, mapping from (-1, 1) to (300, 300)
        mapping = lambda x: ((self.image_resolution-1)/2) * (x+1)
        good_x, good_y = np.where(mask == 255)
        
        for i in range(1, self.patch_sample_tries+1):
            sampled_point = np.random.multivariate_normal(
                self.patch_sample_mean, self.patch_sample_cov, size=(1)
            ).flatten()
            x = round(
                np.clip(mapping(sampled_point[0]),
                        *self.patch_sample_range).item()
            )
            y = round(
                np.clip(mapping(sampled_point[1]), 
                        *self.patch_sample_range).item()
            )
            
            if x in good_x and y in good_y:
            #print("Sampling ran for {} loops".format(i))
                # patch = image[
                #     y-patch_resolution:y+patch_resolution, 
                #     x-patch_resolution:x+patch_resolution
                # ]
                # white_ratio = (patch==255).sum() / patch_mask.size
                # if white_ratio <= self.patch_sample_white_cutoff:
                #     break
                patch_mask = mask[
                    y-patch_resolution:y+patch_resolution, 
                    x-patch_resolution:x+patch_resolution
                ]
                white_ratio = (patch_mask==0).sum() / patch_mask.size
                if white_ratio <= self.patch_sample_white_cutoff:
                    break
        
        patch = image[
            y-patch_resolution:y+patch_resolution, 
            x-patch_resolution:x+patch_resolution
        ]
        size = (self.patch_resize_resolution, self.patch_resize_resolution)
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_CUBIC)
        
        return patch, i
    
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

        patch_image, iters = self.get_patch_image(image, item_mask)
        patch_color_hist = get_color_histogram(
            patch_image, None, self.hist_bin_size
        )
        patch_color_hist = self.hist_transform(patch_color_hist)

        sample = {
            'image': self._transform(image),
            'patch_image': self._transform(patch_image),
            'global_color_hist': global_color_hist,
            'patch_color_hist': patch_color_hist,
            'index': idx,
            'image_name': self.image_ids[idx]
            # 'iters': iters,
            # 'mask': item_mask
        }
        
        return sample