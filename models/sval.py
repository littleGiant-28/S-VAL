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
        
        self.ckpt_list = deque([])
        self.start_epoch = 0
        
        self.grad_accum_steps = (
            self.config.train.batch_size // self.config.train.grad_accum_batch_size
        )
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
            
            slpd_memory_size = (no_images, self.config.model.projection_dim)
            td_memory_size = (no_images, self.config.model.projection_dim, 
                              self.config.model.projection_dim)
            self.slpd_bank = SLPDMemoryBank(
                slpd_memory_size, self.config.train.bank_momentum_eta, self.logger
            ).to(self.device)
            self.td_bank = TDMemoryBank(
                td_memory_size, self.config.train.bank_momentum_eta, self.logger
            ).to(self.device)

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
        
    def save_model(self, current_epoch):
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
            'sval': self.feature_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'slpd_bank': self.slpd_bank.state_dict(),
            'td_bank': self.td_bank.state_dict()
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
            self.slpd_bank.load_state_dict(ckpt['slpd_bank'])
            self.td_bank.load_state_dict(ckpt['td_bank'])
        else:
            self.logger.info("Using weights as pertrained model")
            
    def initalize_memory(self, dataloader):
        self.slpd_bank.initialize_memory(
            self.feature_model, dataloader, self.device
        )
        self.td_bank.initialize_memory(
            self.feature_model, dataloader, self.device
        )
        
    def memory_index_reset(self):
        self.slpd_bank.reset_after_epoch()
        self.td_bank.reset_after_epoch()
            
    def step(self, batch, current_step, is_last_step):
        #TODO: Add checks for nan, inf; Check targets again
        image = batch['image'].to(self.device)
        patch_image = batch['patch_image'].to(self.device)
        global_color_hist = batch['global_color_hist'].to(self.device)
        patch_color_hist = batch['patch_color_hist'].to(self.device)
        indices = batch['indices'].to(self.device)
        
        batch_size = image.shape[0]
        stats = {}
        
        self.feature_model.train()
        p_global_color_hist, p_texture_features, \
            p_slp_features, p_patch_color_hist \
                = self.feature_model(image, patch_image)
            
        hist_loss0 = F.kl_div(
            global_color_hist, p_global_color_hist, reduction='none'
        ).sum(dim=2).mean()
        hist_loss1 = F.kl_div(
            patch_color_hist, p_patch_color_hist, reduction='none'
        ).sum(dim=2).mean()
        hist_loss = (hist_loss0 + hist_loss1) / 2

        logit_slpd_vector = torch.matmul(
            p_slp_features, self.slpd_bank.memory.T
        ) / self.config.train.temperature
        slpd_loss = F.cross_entropy(logit_slpd_vector, indices)
        
        logit_td_vector = torch.matmul(
            p_texture_features.view(batch_size, -1), 
            self.td_bank.memory.view(self.td_bank.memory_size[0], -1).T
        ) / self.config.train.temperature
        td_loss = F.cross_entropy(logit_td_vector, indices)
        
        total_loss = self.config.train.lambda_rgb * hist_loss + \
            self.config.train.lambda_slpd * slpd_loss + \
            self.config.train.lambda_td * td_loss
            
        total_loss.backward()
        if ((current_step+1) % self.grad_accum_steps == 0) or is_last_step:
            self.optimizer.step()
        
        self.slpd_bank.update(p_slp_features.detach(), indices)
        self.td_bank.update(p_texture_features.detach(), indices)
        
        #logging
        stats['hist_loss'] = hist_loss.item()
        stats['slpd_loss'] = slpd_loss.item()
        stats['td_loss'] = td_loss.item()
        stats['total_loss'] = total_loss.item()
        
        self.tfb_logger.log_scalars(
            current_step, list(stats.keys()), list(stats.values())
        )
        self.logger.info(
            "Step {}: Hist loss: {} SLPD Loss: {} TD Loss: {} Total Loss: {}"
            .format(current_step, stats['hist_loss'], stats['slpd_loss'],
                    stats['td_loss'], stats['total_loss'])
        )
        
        return stats
        
            
        