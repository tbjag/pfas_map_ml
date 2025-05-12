import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim import Adam

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        # Validation metrics only
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_prec = torchmetrics.Precision(task="binary")
        self.val_rec = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

        # Individual conv layers
        self.conv1 = self.conv_block(in_channels=52, out_channels=256)
        self.conv2 = self.conv_block(in_channels=256, out_channels=128)
        self.conv3 = self.conv_block(in_channels=128, out_channels=64)
        self.conv4 = self.conv_block(in_channels=64, out_channels=32)
        self.conv5 = self.conv_block(in_channels=32, out_channels=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.sigmoid(x)

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

    def on_validation_epoch_end(self):
        # Log validation metrics
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