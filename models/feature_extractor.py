import torch
import torchvision
import torch.nn.functional as F

def gram_matrix(x):
    x = x.unsqueeze(2)  #HW->1
    x_t = x.transpose(1, 2)
    gm = x.bmm(x_t)
    
    return gm

class ColorHistogramHead(torch.nn.Module):
    
    def __init__(self, in_features, config, **kwargs):
        super(ColorHistogramHead, self).__init__()
        
        hist_size = 256 // config.model.hist_bin_size
        p_dim = config.model.projection_dim
        self.projection_fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, p_dim),
            torch.nn.ReLU()
        )
        self.hist_fc = torch.nn.Sequential(
            torch.nn.Linear(p_dim, 3*hist_size) #3 for RGB
        )
        
    def forward(self, features):
        b, _ = features.shape
        x = self.projection_fc(features)
        x = self.hist_fc(x)
        x = x.reshape(b, 3, -1)
        x = F.normalize(x, p=2, dim=-1)
        hist_probs = F.softmax(x, dim=-1)
        
        return hist_probs
    
class TextureDiscriminationHead(torch.nn.Module):
    
    def __init__(self, in_features, config, **kwargs):
        super(TextureDiscriminationHead, self).__init__()
        
        p_dim = config.model.projection_dim
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features, p_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(p_dim, p_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, features):
        x = self.projection_head(features)
        x = F.normalize(x, p=2, dim=-1)
        G = gram_matrix(x)
        
        return G
    
class LocalPatchDiscriminatorHead(torch.nn.Module):
    
    def __init__(self, in_features, config, **kwargs):
        super(LocalPatchDiscriminatorHead, self).__init__()
        
        p_dim = config.model.projection_dim
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features, p_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(p_dim, p_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, features):
        x = self.projection_head(features)
        x = F.normalize(x, p=2, dim=-1)
        return x
        

class FeatureExtractor(torch.nn.Module):
    supported_backbones = {
        "resnet50": torchvision.models.resnet50
    }
    feature_sizes = {
        "resnet50": 2048
    }

    def __init__(self, config, logger):
        super(FeatureExtractor, self).__init__()
        backbone_name = config.model.backbone
        if  not backbone_name in self.supported_backbones:
            msg = "Provided backbone is not suppoerted: {}".format(backbone_name)
            logger.info(msg)
            raise Exception(msg)

        self.backbone, feature_size = self.get_backbone(config, backbone_name)
        self.colorhist_head = ColorHistogramHead(feature_size, config)
        self.td_head = TextureDiscriminationHead(feature_size, config)
        self.slpd_head = LocalPatchDiscriminatorHead(feature_size, config)
        
        
    def get_backbone(self, config, backbone_name):
        """
        Create backbone, remove last fc layer from backbone and return it
        """
        backbone = self.supported_backbones[backbone_name](
            pretrained=config.model.backbone_pretrained
        )
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        return backbone, self.feature_sizes[config.model.backbone]
    
    def get_slpd_features(self, image_patch):
        patch_features = self.backbone(image_patch)
        patch_features = patch_features.reshape(-1, patch_features.shape[1])
        slpd_features = self.slpd_head(patch_features)
        
        return slpd_features
    
    def get_td_features(self, image):
        features = self.backbone(image)
        features = features.reshape(-1, features.shape[1])
        td_features = self.td_head(features)
        
        return td_features
    
    def forward(self, image, image_patch):
        features = self.backbone(image)
        
        features = features.reshape(-1, features.shape[1])
        global_hist_probs = self.colorhist_head(features)
        texture_features = self.td_head(features)
        
        patch_features = self.backbone(image_patch)
        patch_features = patch_features.reshape(-1, patch_features.shape[1])
        patch_hist_probs = self.colorhist_head(patch_features)
        slp_features = self.slpd_head(patch_features)
        
        return global_hist_probs, texture_features, \
            slp_features, patch_hist_probs