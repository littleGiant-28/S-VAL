import torch
from tqdm import tqdm

class BaseMemory(torch.nn.Module):
    
    def __init__(self, memory_size, update_weight, logger):
        super(BaseMemory, self).__init__()
        
        self.memory_size = memory_size
        self.update_weight = update_weight
        self.logger = logger
        self.index_update = []
        bank = torch.empty(self.memory_size, requires_grad=False)
        self.register_buffer('memory', bank)
        
    def initialize_memory(self, model, data_loader):
        raise NotImplementedError
    
    def reset_after_epoch(self):
        msg = ("Updated index length: {}, Memory length: {}"
               .format(len(self.index_update), len(self.memory)))
        assert len(self.index_update) == len(self.memory), \
            self.logger.info(msg)
        self.index_update = []
        
    def reset(self):
        self.memory = torch.empty(self.memory_size, requires_grad=False)
        
    def check_for_double_update(self, indices):
        for index in indices:
            if index in self.index_update:
                msg = "Index {} to update is already updated.".format(index)
                self.logger.info(msg)
                raise Exception(msg)
    
    def update(self, tensor, indices):
        self.check_for_double_update(indices.cpu().tolist())
        self.index_update.extend(indices.cpu().tolist())
        #indices = torch.LongTensor(indices, device=tensor.device)
        old_memory = torch.index_select(self.memory, dim=0, index=indices)
        new_memory = (1-self.update_weight)*old_memory + \
                        self.update_weight*tensor
        self.memory.index_copy_(dim=0, index=indices, source=new_memory)
        
    def forward(self):
        raise NotImplementedError
        
class SLPDMemoryBank(BaseMemory):
    
    def __init__(self, memory_size, update_weight, logger):
        super(SLPDMemoryBank, self).__init__(memory_size, update_weight, logger)
        
    def initialize_memory(self, model, dataloader, device):
        self.logger.info("Initializing SLPD Memory Bank")

        with torch.no_grad():
            model.eval()
            tepoch = tqdm(iter(dataloader), unit='batch')
            for batch in tepoch:
                image_patch = batch['patch_image'].to(device)
                # indices = torch.LongTensor(
                #     batch['indices'], device=image_patch.device
                # )
                indices = batch['indices'].to(device)
                self.check_for_double_update(indices.cpu().tolist())
                self.index_update.extend(indices.cpu().tolist())
                
                slpd_features = model.get_slpd_features(image_patch)
                self.memory.index_copy_(
                    dim=0, index=indices, source=slpd_features
                )
            
            self.reset_after_epoch()
                
class TDMemoryBank(BaseMemory):
    
    def __init__(self, memory_size, update_weight, logger):
        super(TDMemoryBank, self).__init__(memory_size, update_weight, logger)
        
    def initialize_memory(self, model, dataloader, device):
        self.logger.info("Initializing TD Memory Bank")

        with torch.no_grad():
            model.eval()
            tepoch = tqdm(dataloader, unit='batch')
            for batch in tepoch:
                image_patch = batch['image'].to(device)
                # indices = torch.LongTensor(
                #     batch['indices'], device=image_patch.device
                # )
                indices = batch['indices'].to(device)
                self.check_for_double_update(indices.cpu().tolist())
                self.index_update.extend(indices.cpu().tolist())
                
                td_features = model.get_td_features(image_patch)
                self.memory.index_copy_(
                    dim=0, index=indices, source=td_features
                )
                
            self.reset_after_epoch()
                
                