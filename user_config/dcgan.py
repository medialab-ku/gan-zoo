import torch

from core.config import Config
from core.tags import *
from model.dcgan.celeba_dataset import CelebAdataset
from model.dcgan.discriminator import Discriminator
from model.dcgan.generator import Generator
from model.dcgan.trainer import Trainer
from model.dcgan.validator import Validator

gen_param = {
    'num_latent_z': 100,
    'num_features': 64,
    'num_img_channels': 3,

}

disc_param = {
    'num_disc_features': 64,
    'num_img_channels': 3,
}

_optim = torch.optim.Adam
_optim_param = {
    'lr': 0.0002,
    'betas': (0.5, 0.999),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_multi_gpu = False


class GenConfig(Config):
    def __init__(self):
        super().__init__(
            # important keys
            name=__name__,
            tag=GEN,
            model=Generator,
            model_param=gen_param,
            optim=_optim,
            optim_param=_optim_param,
            device=device,

            # training keys
            dataset=[
                CelebAdataset
            ],
            dataset_param={
                'root': './dataset/celeba',
                'image_size': 64
            },
            trainer=Trainer,
            validator=Validator,
            n_batch=128,
            n_worker=8,
            epoch=5,
            multi_gpu=_multi_gpu,

            # current settings
            num_visual_img=64,
            num_iter_to_save=100,
            you_can_add_anything=':)',
        )


class DiscConfig(Config):
    def __init__(self):
        super().__init__(
            name=__name__,
            tag=DISC,
            model=Discriminator,
            model_param=disc_param,
            optim=_optim,
            optim_param=_optim_param,
            device=device,
        )


class EvalConfig(Config):
    def __init__(self):
        super().__init__(
            name=__name__,
            tag=EVAL,
            model=Generator,
            model_param=gen_param,
            device=device,
        )
