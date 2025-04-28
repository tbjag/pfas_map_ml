# Unet3
# UNet with 3 encoder and decoder blocks, channel reaches size 256
# Input channel is downsized to 32

import torch
import torch.nn as nn
import torch.optim as optim
from models.unet_parts import *
from pytorch_lightning import LightningModule
from torch.optim import Adam

class UNet3(LightningModule):
    def __init__(self, n_channels=50, n_classes=1, dropout_rate=0.15):
        super(UNet3, self).__init__()
        self.best_model_val_loss = float("inf")

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
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Get the aggregated validation loss for the epoch
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        
        # Update best metrics if the current epoch's loss is better
        if self.best_model_val_loss < 0 or val_loss < self.best_model_val_loss:
            self.best_model_val_loss = val_loss
            
            # Log the best metrics
        self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer