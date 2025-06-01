import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torchmetrics import R2Score
from torchmetrics import MeanAbsoluteError as MAE
import torch

class ResNetModel(LightningModule):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.r2_score = R2Score()
        self.mae = MAE()

        self.conv1 = nn.Sequential(
            nn.Conv2d(52, 1024, 3, padding='same'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Residual blocks with channel reduction
        self.res1 = ResidualBlock(1024, 512)
        self.res2 = ResidualBlock(512, 256)
        self.res3 = ResidualBlock(256, 128)
        self.res4 = ResidualBlock(128, 64)
        self.res5 = ResidualBlock(64, 32)
        self.res6 = ResidualBlock(32, 16)

        # Final output layer
        self.conv_final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        return self.conv_final(x)

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
        return loss

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

class ResidualBlock(nn.Module):
    """ResNet-style residual block with channel reduction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Shortcut connection for channel mismatch
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, padding='same'),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)