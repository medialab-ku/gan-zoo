import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 num_disc_features,
                 num_img_channels,
                 ):
        super(Discriminator, self).__init__()

        self.num_disc_features = num_disc_features
        self.num_img_channels = num_img_channels

        self.layers = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_img_channels, num_disc_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_disc_features, num_disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_disc_features * 2, num_disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_disc_features * 4, num_disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
