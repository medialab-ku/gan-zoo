import torch
import torchvision.utils as vutils

from core.abstract.abcvalidator import ABCValidator


class Validator(ABCValidator):
    def __init__(self, config, model, writer):
        super().__init__(config, model, writer)

        self.gen = model

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(self.config.num_visual_img, self.gen.num_latent_z, 1, 1,
                                       device=self.config.device)

    def step(self, epoch, step):
        results = {
            'ssid': 0,
            'ppim': 0,
        }

        self.gen.eval()

        with torch.no_grad():
            fake_img = self.gen(self.fixed_noise).detach()
            norm_fake_imgs = vutils.make_grid(fake_img, padding=2, normalize=True)

        self.writer.add_image('valid/fake_img', norm_fake_imgs, step)

        return results, 0
