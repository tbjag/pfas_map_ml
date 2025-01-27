# Unet2
# UNet with 3 encoder and decoder blocks, channel reaches size 256
# Input channel is downsized to 32

import torch
import torch.nn as nn
import torch.optim as optim
from models.unet_parts import *

class UNet3(nn.Module):
    def __init__(self, n_channels=263, n_classes=1, dropout_rate=0.25):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout_rate)

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.up1 = (Up(256, 128))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(64, 32))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x = self.up1(x4, x3)
        x = self.dropout(x)
        x = self.up2(x, x2)
        x = self.dropout(x)
        x = self.up3(x, x1)
        x = self.dropout(x)
        x = self.outc(x)
        return x