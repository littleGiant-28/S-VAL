from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN()
_C.train = CN()
_C.model = CN()
_C.general = CN()
_C.logging = CN()
_C.save_load = CN()

_C.general.device = 'gpu'
_C.general.device_id = 0
_C.general.is_training = True
_C.general.seed = 22
_C.general.notes = """Base Experiment"""

_C.data.image_root = "data/polyvore_outfits/images"
_C.data.item_json_path = "data/polyvore_outfits/polyvore_item_metadata.json"
_C.data.image_extension = ".jpg"
_C.data.image_resolution = 300
_C.data.patch_resolution_range = [0.05, 0.15]
_C.data.patch_resize_resolution = 0.1
_C.data.patch_sample_mean = 0
_C.data.patch_sample_cov = 0.2
_C.data.patch_sample_tries = 30
_C.data.patch_sample_range = [75, 225]
_C.data.patch_sample_white_cutoff = 0.6
_C.data.num_workers = 2
_C.data.prefetch_factor = 2
_C.data.stability_constant = 1e-8

_C.save_load.exp_root = 'experiments'
_C.save_load.exp_name = 'experiment_1'
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
_C.train.lambda_td = 1e-5
_C.train.temperature = 0.07
_C.train.bank_momentum_eta = 0.5
_C.train.betas=(0.9, 0.999)
_C.train.eps=1e-08
_C.train.grad_accum_batch_size = 32

_C.model.backbone = 'resnet50'
_C.model.backbone_pretrained = True
_C.model.projection_dim = 128
_C.model.hist_bin_size = 10

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()