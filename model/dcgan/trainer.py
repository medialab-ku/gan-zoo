from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from core.abstract.abctrainer import ABCTrainer


class Trainer(ABCTrainer):
    def __init__(self, config, model, optim, writer):
        super().__init__(config, model, optim, writer)

        self.gen = model['gen']
        self.disc = model['disc']
        self.gen_optim = optim['gen']
        self.disc_optim = optim['disc']

        self.gen.to(self.config.device)
        self.disc.to(self.config.device)

        self.bce = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(self.config.num_visual_img, self.gen.num_latent_z, 1, 1,
                                       device=self.config.device)

    def step(self, epoch, step):
        postfix = OrderedDict({
            'errD': 0,
            'errG': 0,
        })

        device = self.config.device
        self.gen.train()
        self.disc.train()

        with tqdm(total=len(self.data_loader), desc='{} epoch'.format(epoch)) as progress:
            for data_ in self.data_loader:
                # Prepare training data
                real_img = data_[0].to(device)
                batch_size = real_img.size(0)
                noise = torch.randn(batch_size, self.gen.num_latent_z, 1, 1,
                                    device=device)  # Generate batch of latent vectors
                fake_img = self.gen(noise)  # Generate fake image batch with G
                real_label = torch.full((batch_size,), self.real_label, device=device)
                fake_label = torch.full_like(real_label, self.fake_label)

                # train models
                errD = self.train_discriminator(step, epoch,
                                                real_img, real_label,
                                                fake_img, fake_label)
                errG = self.train_generator(step, epoch,
                                            fake_img, real_label)

                # finish iteration
                postfix['errD'] = errD.item()
                postfix['errG'] = errG.item()

                progress.set_postfix(ordered_dict=postfix)
                progress.update(1)
                step = step + 1

        epoch = epoch + 1

        return epoch, step, errG

    def train_discriminator(self,
                            step, epoch,
                            real_img, real_label,
                            fake_img, fake_label):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        self.disc.zero_grad()
        errD_real, D_x = calc_model(self.disc, self.bce,
                                    real_img, real_label)  # Train with all-real batch

        errD_fake, D_G_z1 = calc_model(self.disc, self.bce,
                                       fake_img.detach(), fake_label)  # Train with all-fake batch

        errD = errD_real + errD_fake

        self.disc_optim.step()

        self.writer.add_scalar('train/D/errD_real', errD_real, step)
        self.writer.add_scalar('train/D/errD_fake', errD_fake, step)
        self.writer.add_scalar('train/D/errD', errD, step)
        self.writer.add_scalar('train/D/D_x', D_x, step)
        self.writer.add_scalar('train/D/D_G_z1', D_G_z1, step)
        if step % self.config.num_iter_to_save == 0:
            norm_real_imgs = vutils.make_grid(real_img[:self.config.num_visual_img],
                                              padding=2, normalize=True)

            with torch.no_grad():
                fake_img = self.gen(self.fixed_noise)
                norm_fake_imgs = vutils.make_grid(fake_img, padding=2, normalize=True)

            self.writer.add_image('train/D/real_img', norm_real_imgs, step)
            self.writer.add_image('train/D/fake_img', norm_fake_imgs, step)

        return errD

    def train_generator(self,
                        step, epoch,
                        fake_img, real_label):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.gen.zero_grad()

        errG, D_G_z2 = calc_model(self.disc, self.bce,
                                  fake_img, real_label)

        self.gen_optim.step()

        self.writer.add_scalar('train/G/errG', errG, step)
        self.writer.add_scalar('train/G/D_G_z2', D_G_z2, step)

        return errG


def calc_model(model, criterion, img, label):
    output = model(img).view(-1)  # Forward pass real batch through D
    err = criterion(output, label)  # Calculate loss on all-real batch
    err.backward()  # Calculate gradients for D in backward pass

    return err, output.mean().item()
