import os
import torch
from torch.utils.tensorboard import SummaryWriter

class TFBLogger(object):

    def __init__(self, cfg):
        self.save_dir = cfg.logging.tfb_log_dir
        self.log_interval = cfg.logging.tfb_log_interval
        self.log_gradient = cfg.logging.tfb_log_gradient
        self.gradient_interval = cfg.logging.tfb_log_gradient_interval
        self.initialize_logger()

    def initialize_logger(self):
        self.tfb_writer = SummaryWriter(self.save_dir)

    def close(self):
        self.tfb_writer.close()

    def log_scalars(self, iter_no, tags, values):
        if iter_no % self.log_interval == 0:
            if isinstance(values, torch.Tensor): 
                values = values.detach().cpu().numpy()
            assert len(tags) == len(values)

            for tag, value in zip(tags, values):
                if isinstance(value, dict):
                    self.tfb_writer.add_scalars(tag, value, global_step=iter_no)
                else:
                    self.tfb_writer.add_scalar(
                        tag, float(value), global_step=iter_no
                    )

            self.tfb_writer.flush()

    def log_params_gradients(self, iter_no, model, is_distributed=False):
        if self.log_gradient and iter_no % self.gradient_interval == 0:
            if is_distributed: model = model.module

            with torch.no_grad():
                for i,k in model.named_parameters():
                    self.tfb_writer.add_histogram(
                        'param/{}'.format(i), k.detach(), global_step=iter_no
                    )
                    self.tfb_writer.add_histogram(
                        'grad/{}'.format(i), k.grad.detach(),
                        global_step=iter_no
                    )

            self.tfb_writer.flush()
