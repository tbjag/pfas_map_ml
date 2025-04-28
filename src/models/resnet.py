import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam

class ResNetModel(LightningModule):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.best_model_val_loss = -1  # Track best validation loss

        # Initial convolution block (36 -> 256 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(36, 256, 3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Residual blocks with channel reduction
        self.res1 = ResidualBlock(256, 128)  # 256 -> 128
        self.res2 = ResidualBlock(128, 64)    # 128 -> 64
        self.res3 = ResidualBlock(64, 32)     # 64 -> 32
        self.res4 = ResidualBlock(32, 16)     # 32 -> 16

        # Final output layer
        self.conv_final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.conv_final(x)

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
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        if self.best_model_val_loss < 0 or val_loss < self.best_model_val_loss:
            self.best_model_val_loss = val_loss
        self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

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