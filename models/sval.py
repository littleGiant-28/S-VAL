import os
from collections import deque

import torch
import numpy as np
import torch.nn.functional as F

from .feature_extractor import FeatureExtractor
from utils import SLPDMemoryBank, TDMemoryBank

def get_tgn(model):
    tgn = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.grad.detach().data.norm(2).item()
            tgn = tgn + (param_norm ** 2)
        else:
            print(name)

    tgn = tgn ** 0.5

    return tgn

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
                eps=self.config.train.eps,
                weight_decay=self.config.train.weight_decay
            )

            slpd_memory_size = (no_images, self.config.model.projection_dim)
            td_memory_size = (no_images, self.config.model.projection_dim * \
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
            self.total_steps = ckpt['total_steps']
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
            
    def step(self, batch, current_epoch, current_step, is_last_step, total_steps):
        #TODO: Add checks for nan, inf; Check targets again
        image = batch['image'].to(self.device)
        patch_image = batch['patch_image'].to(self.device)
        global_color_hist = batch['global_color_hist'].to(self.device)
        patch_color_hist = batch['patch_color_hist'].to(self.device)
        indices = batch['indices'].to(self.device)

        stats = {}
        
        self.feature_model.train()
        p_global_color_hist, p_texture_features, \
            p_slp_features, p_patch_color_hist \
                = self.feature_model(image, patch_image)

        hist_loss_global = p_global_color_hist * \
                            (p_global_color_hist.log() - global_color_hist)
        hist_loss_global = hist_loss_global.sum(dim=(1, 2)).mean()
        hist_loss_local = p_patch_color_hist * \
                            (p_patch_color_hist.log() - patch_color_hist)
        hist_loss_local = hist_loss_local.sum(dim=(1,2)).mean()
        hist_loss = (hist_loss_global + hist_loss_local) / 2

        if torch.any(torch.isnan(hist_loss.detach())) or \
            torch.any(torch.isinf(hist_loss.detach())):
            raise Exception("Hist loss is nan or inf: ", hist_loss)

        logit_slpd_vector = torch.matmul(
            p_slp_features, self.slpd_bank.memory.T
        ) / self.config.train.temperature
        slpd_loss = F.cross_entropy(logit_slpd_vector, indices)

        if torch.any(torch.isnan(slpd_loss.detach())) or \
            torch.any(torch.isinf(slpd_loss.detach())):
            raise Exception("SLPD loss is nan or inf: ", slpd_loss)

        logit_td_vector = torch.matmul(
            p_texture_features, 
            self.td_bank.memory.T
        ) / self.config.train.temperature
        td_loss = F.cross_entropy(logit_td_vector, indices)

        if torch.any(torch.isnan(td_loss.detach())) or \
            torch.any(torch.isinf(td_loss.detach())):
            raise Exception("TD loss is nan or inf: ", td_loss)

        total_loss = self.config.train.lambda_rgb * hist_loss + \
            self.config.train.lambda_slpd * slpd_loss + \
            self.config.train.lambda_td * td_loss  

        total_loss = total_loss / (self.grad_accum_steps)
        total_loss.backward()
        
        #logging
        stats['hist_loss'] = hist_loss.item()
        stats['hist_loss_local'] = hist_loss_local.item()
        stats['hist_loss_global'] = hist_loss_global.item()
        stats['slpd_loss'] = slpd_loss.item()
        stats['td_loss'] = td_loss.item()
        stats['total_loss'] = total_loss.item()
        stats['tgn'] = get_tgn(self.feature_model)

        self.tfb_logger.log_scalars(
            total_steps, list(stats.keys()), list(stats.values()),
            prefix="train/"
        )
        self.tfb_logger.log_params_gradients(total_steps, self.feature_model)

        if ((total_steps+1) % self.grad_accum_steps == 0) or is_last_step:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.td_bank.update(p_texture_features.detach(), indices)
        self.slpd_bank.update(p_slp_features.detach(), indices)

        # self.logger.info(
        #     "Epoch: {} Step {}: Hist loss: {} SLPD Loss: {} TD Loss: {} Total Loss: {}"
        #     .format(current_epoch, current_step, stats['hist_loss'], stats['slpd_loss'],
        #             stats['td_loss'], stats['total_loss'])
        # )

        # stats.pop('hist_loss_local')
        # stats.pop('hist_loss_global')
        
        return stats

    def test_step(self, batch):
        with torch.no_grad():
            image = batch['image'].to(self.device)
            patch_image = batch['patch_image'].to(self.device)
            global_color_hist = batch['global_color_hist'].to(self.device)
            patch_color_hist = batch['patch_color_hist'].to(self.device)
            indices = batch['indices'].to(self.device)

            stats = {}

            self.feature_model.eval()

            p_global_color_hist, p_texture_features, \
                p_slp_features, p_patch_color_hist \
                    = self.feature_model(image, patch_image)

            hist_loss_global = p_global_color_hist * \
                                (p_global_color_hist.log() - global_color_hist)
            hist_loss_global = hist_loss_global.sum(dim=(1, 2)).mean()
            hist_loss_local = p_patch_color_hist * \
                                (p_patch_color_hist.log() - patch_color_hist)
            hist_loss_local = hist_loss_local.sum(dim=(1,2)).mean()
            hist_loss = (hist_loss_global + hist_loss_local) / 2

            if torch.any(torch.isnan(hist_loss.detach())) or \
                torch.any(torch.isinf(hist_loss.detach())):
                raise Exception("Hist loss is nan or inf: ", hist_loss)

            logit_slpd_vector = torch.matmul(
                p_slp_features, self.slpd_bank.memory.T
            ) / self.config.train.temperature
            slpd_loss = F.cross_entropy(logit_slpd_vector, indices)

            if torch.any(torch.isnan(slpd_loss.detach())) or \
                torch.any(torch.isinf(slpd_loss.detach())):
                raise Exception("SLPD loss is nan or inf: ", slpd_loss)

            logit_td_vector = torch.matmul(
                p_texture_features, 
                self.td_bank.memory.T
            ) / self.config.train.temperature
            td_loss = F.cross_entropy(logit_td_vector, indices)

            if torch.any(torch.isnan(td_loss.detach())) or \
                torch.any(torch.isinf(td_loss.detach())):
                raise Exception("TD loss is nan or inf: ", td_loss)

            total_loss = self.config.train.lambda_rgb * hist_loss + \
                self.config.train.lambda_slpd * slpd_loss + \
                self.config.train.lambda_td * td_loss  

            #logging
            stats['hist_loss'] = hist_loss.item()
            stats['hist_loss_local'] = hist_loss_local.item()
            stats['hist_loss_global'] = hist_loss_global.item()
            stats['slpd_loss'] = slpd_loss.item()
            stats['td_loss'] = td_loss.item()
            stats['total_loss'] = total_loss.item()

        return stats
        
            
        