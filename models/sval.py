import os
from collections import deque

import torch
import numpy as np

from .feature_extractor import FeatureExtractor
from utils import SLPDMemoryBank, TDMemoryBank

class SVAL(object):
    
    def __init__(self, config, logger, tfb_logger, no_images):
        self.config = config
        self.logger = logger
        self.tfb_logger = tfb_logger
        self.device = self.get_device()
        self.is_training = self.config.general.is_training
        self.start_step = 0
        
        self.feature_model = FeatureExtractor(
            self.config, self.logger
        ).to(self.deivce)
        
        self.ckpt_list = deque([])
        self.start_step = 0
        
        if self.is_training:
            self.logger.info("Creating optimizer for training")
            self.optimizer = torch.optim.Adam(
                list(self.feature_model.parameters()),
                lr=self.config.train.gen_lr,
                betas=self.config.train.betas,
                eps=self.config.train.eps
            )
            
            slpd_memory_size = (no_images, self.config.model.projection_dim)
            td_memory_size = (no_images, self.config.model.projection_dim, 
                              self.config.model.projection_dim)
            self.slpd_bank = SLPDMemoryBank(
                slpd_memory_size, self.config.train.bank_momentum_eta, self.logger
            )
            self.td_bank = TDMemoryBank(
                td_memory_size, self.config.train.bank_momentum_eta, self.logger
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
        
    def save_model(self, current_step):
        weight_name = 'sval-{}.pth'.format(current_step)
        save_path = os.path.join(
            self.config.save_load.exp_path, weight_name
        )
        self.ckpt_list.append(weight_name)
        self.logger.info(
            "Saving weights for current step at {}".format(save_path)
        )

        torch.save({
            'current_step': current_step,
            'sval': self.feature_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'slpd_bank': self.slpd_bank.state_dict(),
            'td_bank': self.td_bank.state_dict()
        }, save_path)

        if len(self.ckpt_list) > self.config.keep_last_ckpts:
            delete_ckpt = self.ckpt_list.popleft()
            delete_ckpt = os.path.join(
                self.config.exp_path, delete_ckpt
            )
            self.logger.info(
                "Removing old weight file at {}".format(delete_ckpt)
            )
            os.remove(delete_ckpt)

    def load_model(self):
        self.logger.info(
            "Loading weights from {}".format(self.config.save_load.load_path)
        )
        ckpt = torch.laod(
            self.config.save_load.load_path, map_location=self.device
        )
        self.feature_model.load_state_dict(ckpt['sval'])

        if not self.config.save_load.pretrained:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_step = ckpt['current_step']
            self.slpd_bank.load_state_dict(ckpt['slpd_bank'])
            self.td_bank.load_state_dict(ckpt['td_bank'])
        else:
            self.logger.info("Using weights as pertrained model")
            
    def step(self, ):
        pass