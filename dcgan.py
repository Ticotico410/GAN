import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    DCGAN generator (upsample+conv).
    Output in [-1, 1] via Tanh.
    """
    def __init__(self, img_size=64, channels=3, latent_dim=100):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),  # 16->32
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 32->64
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """
    DCGAN discriminator.
    """
    def __init__(self, img_size=64, channels=3):
        super().__init__()
        self.img_size = img_size
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # 64->32
            *discriminator_block(16, 32),                 # 32->16
            *discriminator_block(32, 64),                 # 16->8
            *discriminator_block(64, 128),                # 8->4
        )

        ds_size = img_size // 2 ** 4  # 64->4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity