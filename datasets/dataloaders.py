import torch
import numpy as np
from .polyvore_dataset import PolyvoreDataset

def polyvore_collate_fn(batch):
    image_tensor = torch.stack(
        [item['image'] for item in batch], dim=0
    )
    
    global_color_hist_tensor = torch.stack(
        [item['global_color_hist'] for item in batch], dim=0
    )
    
    batch = {
        'image': image_tensor,
        'global_color_hist': global_color_hist_tensor
    }
    
    return batch

def worker_init_fn_for_seed(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_polyvore_dataloader(config, logger):
    polyvore_dataset = PolyvoreDataset(config, logger)
    no_images = len(polyvore_dataset)
    
    dataloader = torch.utils.data.DataLoader(
        polyvore_dataset,
        batch_size=config.train.batch_size,
        collate_fn=polyvore_collate_fn,
        shuffle=True,
        num_workers=config.data.num_workers,
        worker_init_fn=worker_init_fn_for_seed,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=False   
    )
    
    return dataloader, no_images