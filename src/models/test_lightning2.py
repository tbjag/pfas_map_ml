import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.best_model_val_loss = -1

        # Input: 50 channels, Output: 256 channels
        self.conv1 = nn.Conv2d(in_channels=36, out_channels=256, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        # 256 → 128
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        # 128 → 64
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        # 64 → 32
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)

        # 32 → 16
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.2)

        # 16 → 1 (final output, keeping your original idea)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout5(self.relu5(self.bn5(self.conv5(x))))
        x = self.conv6(x)
        return x

    def conv_block(in_channels, out_channels, dropout_rate=0.2):
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
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        if self.best_model_val_loss < 0 or val_loss < self.best_model_val_loss:
            self.best_model_val_loss = val_loss
        self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
