# Unet1
# UNet with 4 encoder and decoder blocks, channel input reaches size 1024
# Input channel is downsized to 64

import torch
import torch.nn as nn
import torch.optim as optim
from models.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels=263, n_classes=1, dropout_rate=0.25):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout_rate)

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
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)
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

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class EnhancedModel(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(EnhancedModel, self).__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(263, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Deep feature processing
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.layer3 = self._make_layer(256, 512)
        self.layer4 = self._make_layer(512, 1024)
        self.layer5 = self._make_layer(1024, 512)
        self.layer6 = self._make_layer(512, 256)
        self.layer7 = self._make_layer(256, 128)
        self.layer8 = self._make_layer(128, 64)
        
        # Additional feature processing with larger receptive field
        self.conv_large = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn_large = nn.BatchNorm2d(64)
        
        # Feature reduction layers
        self.conv_reduce1 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn_reduce1 = nn.BatchNorm2d(32)
        self.conv_reduce2 = nn.Conv2d(32, 16, kernel_size=1)
        self.bn_reduce2 = nn.BatchNorm2d(16)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Deep feature processing
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.layer5(x)
        x = self.dropout(x)
        x = self.layer6(x)
        x = self.dropout(x)
        x = self.layer7(x)
        x = self.dropout(x)
        x = self.layer8(x)
        x = self.dropout(x)
        
        # Additional feature processing
        residual = x
        x = self.conv_large(x)
        x = self.bn_large(x)
        x = self.relu(x)
        x = x + residual  # Skip connection
        
        # Feature reduction
        x = self.conv_reduce1(x)
        x = self.bn_reduce1(x)
        x = self.relu(x)
        x = self.conv_reduce2(x)
        x = self.bn_reduce2(x)
        x = self.relu(x)
        
        # Final prediction
        x = self.final_conv(x)
        return x