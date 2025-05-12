import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torchmetrics import R2Score
import torch

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.best_model_val_loss = float("inf")
        self.best_model_val_r2 = None
        self.r2_score = R2Score()

        self.conv1 = self.conv_block(in_channels=50, out_channels=1024)
        self.conv2 = self.conv_block(in_channels=1024, out_channels=512)
        self.conv3 = self.conv_block(in_channels=512, out_channels=256)
        self.conv4 = self.conv_block(in_channels=256, out_channels=128)
        self.conv5 = self.conv_block(in_channels=128, out_channels=64)
        self.conv6 = self.conv_block(in_channels=64, out_channels=32)

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x

    def conv_block(self, in_channels, out_channels, dropout_rate=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

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
        r2 = self.r2_score(y_pred.flatten(1), y.flatten(1))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_r2", r2, prog_bar=True)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        val_r2 = self.trainer.callback_metrics["val_r2"].item()
        if self.best_model_val_loss < 0 or val_loss < self.best_model_val_loss:
            self.best_model_val_loss = val_loss
            self.best_model_val_r2 = val_r2
        self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)
        self.log("best_model_val_r2", self.best_model_val_r2, prog_bar=True)

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
