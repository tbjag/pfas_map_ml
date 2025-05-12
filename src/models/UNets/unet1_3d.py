import torch
import torch.nn as nn
from models.UNets.unet_parts import *
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torchmetrics import R2Score
from torchmetrics import MeanAbsoluteError as MAE

class UNet1(LightningModule):
    def __init__(self, n_channels=52, n_classes=3, dropout_rate=0.2):
        super(UNet1, self).__init__()
        self.r2_score = R2Score()
        self.mae = MAE()
        
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
        #loss = nn.MSELoss(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.r2_score.update(y_pred.flatten(1), y.flatten(1))  # Accumulate R2 across batches
        self.mae.update(y_pred, y)
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        # Compute R2 over all validation batches
        val_r2 = self.r2_score.compute()
        val_mae = self.mae.compute()
        self.log("val_r2", val_r2, prog_bar=True)
        self.log("val_mae", val_mae, prog_bar=True)
        self.r2_score.reset()  # Reset for next epoch
        self.mae.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #     },
        # }
        return {
            "optimizer": optimizer
        }