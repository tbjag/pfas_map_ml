# Unet1
# UNet with 4 encoder and decoder blocks, channel input reaches size 1024
# Input channel is downsized to 64

import torch
import torch.nn as nn
from models.UNets.unet_parts import *
from pytorch_lightning import LightningModule
from torch.optim import Adam

class UNet1(LightningModule):
    def __init__(self, n_channels=50, n_classes=1, dropout_rate=0.25):
        super(UNet1, self).__init__()
        self.r2_score = R2Score()
        
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout_rate)

        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Keeping the extra down layer from original UNet1

        # Decoder (expanding path)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Forward pass with dropout
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
        
        return self.outc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.r2_score.update(y_pred.flatten(1), y.flatten(1))  # Accumulate R2 across batches
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Compute R2 over all validation batches
        val_r2 = self.r2_score.compute()
        self.log("val_r2", val_r2, prog_bar=True)
        self.r2_score.reset()  # Reset for next epoch

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    


# IGNORE FOR NOW
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