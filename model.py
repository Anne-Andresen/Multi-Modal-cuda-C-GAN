import torch
import torch.nn as nn
from attention import *
import torch
import torch.nn as nn



num_heads = 8
embed_dim = 128
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = CrossAttention(embed_dim, num_heads)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

        self.final_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, struct):
        x = self.attention(x, struct, struct)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(input_size, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(512 * 16 * 16 * 16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x
class GAN(nn.Module):
    """GAN model"""
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def forward(self, x, labels):
        gen_img = self.generator(x)
        return self.discriminator(gen_img, labels)
