import torch
import torch.nn as nn
import torch.optim as optim
from unet_parts import *
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
# Define the UNet model
# class UNet(nn.Module):
#     def conv_block(self, input, in_c, out_c):
#         x = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)(input)
#         x = nn.ReLU(inplace=True)(x)
#         x = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)(x)
#         x = nn.ReLU(inplace=True)(x)
#         return x
    
#     def down_block(self, input, in_c, out_c):
#         x = self.conv_block(input, in_c, out_c)
#         p = nn.MaxPool2d(2)(x)
#         return x,p

#     def up_block(self, input, skip_features, in_c, out_c):
#         x = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)(input)
#         x = torch.cat(x, skip_features)
#         x = self.conv_block(x, out_c, out_c)
#         return x

#     def __init__(self, in_channels=1, out_channels=1):
#         super(UNet, self).__init__()
#         self.final = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         e1, p1 = self.down_block(x, 263, 64)
#         e2, p2 = self.down_block(p1, 64, 128)
#         e3, p3 = self.down_block(p2, 128, 256)
#         e4, p4 = self.down_block(p3, 256, 512)

#         b = self.conv_block(p4, 512, 1024)

#         # Decoder
#         d1 = self.up_block(b, e4, 1024, 512)
#         d2 = self.up_block(d1, e3, 512, 256)
#         d3 = self.up_block(d2, e2, 256, 128)
#         d4 = self.up_block(d3, e1, 128, 64)

#         out = self.final(d4)
#         return out