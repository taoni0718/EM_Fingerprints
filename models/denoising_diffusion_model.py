import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

# Define the U-Net model architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Define the convolutional and upconvolutional blocks
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        # Define the encoding layers
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the middle layers
        self.middle = conv_block(512, 1024)

        # Define the decoding layers
        self.upconv1 = upconv_block(1024, 512)
        self.upconv2 = upconv_block(512, 256)
        self.upconv3 = upconv_block(256, 128)
        self.upconv4 = upconv_block(128, 64)

        self.decoder1 = conv_block(1024, 512)
        self.decoder2 = conv_block(512, 256)
        self.decoder3 = conv_block(256, 128)
        self.decoder4 = conv_block(128, 64)

        # Define the output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        middle = self.middle(self.pool(enc4))

        dec1 = self.decoder1(torch.cat((enc4, self.upconv1(middle)), 1))
        dec2 = self.decoder2(torch.cat((enc3, self.upconv2(dec1)), 1))
        dec3 = self.decoder3(torch.cat((enc2, self.upconv3(dec2)), 1))
        dec4 = self.decoder4(torch.cat((enc1, self.upconv4(dec3)), 1))

        return self.out_conv(dec4)

# class DiffusionModel(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DiffusionModel, self).__init__()
#         self.unet = UNet(in_channels, out_channels)

#     def forward(self, x, noise_std):
#         noise = torch.randn_like(x) * noise_std
#         noisy_image = x + noise
#         return noisy_image

#     def reverse(self, noisy_image):
#         with torch.no_grad():
#             denoised_image = self.unet(noisy_image)
#         return denoised_image

class DiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiffusionModel, self).__init__()
        self.unet = UNet(in_channels, out_channels)

    def forward(self, x, noise_std):
        noise = torch.randn_like(x) * noise_std
        noisy_image = x + noise
        return noisy_image

    def reverse(self, noisy_image):
        denoised_image = self.unet(noisy_image)
        return denoised_image

    def denoise(self, x, noise_std):
        noisy_image = self.forward(x, noise_std)
        denoised_image = self.reverse(noisy_image)
        return denoised_image
