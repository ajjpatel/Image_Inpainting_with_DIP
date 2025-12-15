import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=512):
        super(ContextEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # input: (3) x128x128
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # 64 x64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),             # 128 x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),            # 256 x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),            # 512 x8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),            # 512 x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),            # 512 x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feature_dim, 4, 2, 1),     # 512 x1x1
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 512, 4, 2, 1),  # 512 x2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),          # 512 x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),          # 512 x8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),          # 256 x16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),          # 128 x32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),           # 64 x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1),# 3 x128x128
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # 64 x64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),            # 128 x32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),           # 256 x16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),           # 512 x8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),             # 1 x5x5 -> scalar
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)
