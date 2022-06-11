import os
from collections import deque

import torch
import numpy as np
import torch.nn.functional as F

from .feature_extractor import FeatureExtractor
from utils import SLPDMemoryBank, TDMemoryBank

class SVAL(object):
    
    def __init__(self, config, logger, tfb_logger, no_images):
        self.config = config
        self.logger = logger
        self.tfb_logger = tfb_logger
        self.device = self.get_device()
        self.is_training = self.config.general.is_training
        
        self.feature_model = FeatureExtractor(
            self.config, self.logger
        ).to(self.device)

        if self.config.save_load.load_models:
            self.load_model()
        
        self.ckpt_list = deque([])
        self.start_epoch = 0
        self.total_steps = 0
        
        # self.grad_accum_steps = (
        #     self.config.train.batch_size // self.config.train.grad_accum_batch_size
        # )
        msg = ("Gradient accumulation number of steps: {}"
               .format(self.grad_accum_steps))
        print(msg)
        self.logger.info(msg)

        if self.is_training:
            self.logger.info("Creating optimizer for training")
            self.optimizer = torch.optim.Adam(
                list(self.feature_model.parameters()),
                lr=self.config.train.lr,
                betas=self.config.train.betas,
                eps=self.config.train.eps
            )

    def get_device(self):
        if self.config.general.device == 'cpu':
            return torch.device('cpu')
        elif self.config.general.device == 'gpu':
            assert int(self.config.general.device_id) <= \
                (torch.cuda.device_count() - 1), "Invalid gpu id specfied"
            return torch.device("cuda:{}".format(self.config.general.device_id))
        else:
            msg = ("Invalid device name specified: {}"
                   .format(self.config.general.device))
            raise ValueError(msg)
        
    def save_model(self, current_epoch, total_steps):
        weight_name = 'sval-{}.pth'.format(current_epoch)
        save_path = os.path.join(
            self.config.save_load.exp_path, weight_name
        )
        self.ckpt_list.append(weight_name)
        self.logger.info(
            "Saving weights for current epoch at {}".format(save_path)
        )

        torch.save({
            'current_epoch': current_epoch,
            'total_steps': total_steps,
            'sval': self.feature_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)

        if len(self.ckpt_list) > self.config.save_load.keep_last_ckpts:
            delete_ckpt = self.ckpt_list.popleft()
            delete_ckpt = os.path.join(
                self.config.save_load.exp_path, delete_ckpt
            )
            self.logger.info(
                "Removing old weight file at {}".format(delete_ckpt)
            )
            os.remove(delete_ckpt)

    def load_model(self):
        self.logger.info(
            "Loading weights from {}".format(self.config.save_load.load_path)
        )
        ckpt = torch.load(
            self.config.save_load.load_path, map_location=self.device
        )
        self.feature_model.load_state_dict(ckpt['sval'])

        if not self.config.save_load.pretrained:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt['current_epoch']
            self.total_steps = ckpt['total_steps']
        else:
            self.logger.info("Using weights as pertrained model")
            
    def step(self, batch, current_epoch, current_step, is_last_step, total_steps):
        #TODO: Add checks for nan, inf; Check targets again
        image = batch['image'].to(self.device)
        global_color_hist = batch['global_color_hist'].to(self.device)
        
        batch_size = image.shape[0]
        stats = {}
        
        self.feature_model.train()
        p_global_color_hist = self.feature_model(image)
            
        # hist_loss = F.kl_div(
        #     global_color_hist, p_global_color_hist, reduction='none'
        # ).sum(dim=2).mean()
        hist_loss = p_global_color_hist * (p_global_color_hist.log() - global_color_hist)
        hist_loss = hist_loss.sum(dim=(1,2)).mean()
        
        total_loss = self.config.train.lambda_rgb * hist_loss      
        total_loss.backward()

        #if ((current_step+1) % self.grad_accum_steps == 0) or is_last_step:
        self.optimizer.step()
        
        #logging
        stats['hist_loss'] = hist_loss.item()
        stats['total_loss'] = total_loss.item()

        self.tfb_logger.log_scalars(
            total_steps, list(stats.keys()), list(stats.values())
        )
        
        return stats
        
            
        