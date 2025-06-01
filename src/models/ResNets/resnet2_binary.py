import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
import torchmetrics
import torch.nn.functional as F

class ResNetModel(LightningModule):
    def __init__(self):
        super(ResNetModel, self).__init__()
        # Classification metrics
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_prec = torchmetrics.Precision(task="binary")
        self.val_rec = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

        # Modified architecture for classification
        self.conv1 = nn.Sequential(
            nn.Conv2d(52, 512, 3, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Residual blocks
        self.res1 = ResidualBlock(512, 256)
        self.res2 = ResidualBlock(256, 128)
        self.res3 = ResidualBlock(128, 64)
        self.res4 = ResidualBlock(64, 32)
        self.res5 = ResidualBlock(32, 16)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.float())
        
        # Update metrics
        self.val_acc(y_pred, y)
        self.val_prec(y_pred, y)
        self.val_rec(y_pred, y)
        self.val_f1(y_pred, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Log metrics
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_prec", self.val_prec.compute())
        self.log("val_rec", self.val_rec.compute())
        self.log("val_f1", self.val_f1.compute())
        
        # Reset metrics
        self.val_acc.reset()
        self.val_prec.reset()
        self.val_rec.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return Adam(self.parameters())

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