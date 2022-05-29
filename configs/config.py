from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN()
_C.train = CN()
_C.model = CN()
_C.general = CN()
_C.logging = CN()
_C.save_load = CN()

_C.general.device = 'cpu'
_C.general.device_id = 0
_C.general.is_training = True

_C.save_load.exp_root = 'experiments'
_C.save_load.exp_name = 'test'
_C.save_load.exp_path = ''
_C.save_load.keep_last_ckpts = 5
_C.save_load.save_every = 1000
_C.save_load.override_dir = False
_C.save_load.load_models = False
_C.save_load.pretrained = False
_C.save_load.load_path = ''

_C.logging.tfb_log_dir = 'runs'
_C.logging.tfb_log_interval = 1
_C.logging.tfb_log_gradient = False
_C.logging.tfb_log_gradient_interval = 100
_C.logging.log_file = 'logs.txt'

_C.train.lr = 5e-5
_C.train.epochs = 150
_C.train.batch_size = 256
_C.train.lambda_rgb = 1
_C.train.lambda_slpd = 1e-2
_C.train.tau = 0.07
_C.train.bank_momentum_eta = 0.5
_C.train.lambda_td = 1e-5
_C.train.betas=(0.9, 0.999)
_C.train.eps=1e-08

_C.model.backbone = 'resnet50'
_C.model.backbone_pretrained = True
_C.model.projection_dim = 128
_C.model.hist_bin_size = 10

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()