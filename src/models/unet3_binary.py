import torch
import torch.nn as nn
import torch.optim as optim
from models.unet_parts import *
from pytorch_lightning import LightningModule
from torch.optim import Adam
import torchmetrics

class UNet3(LightningModule):
    def __init__(self, n_channels=54, n_classes=1, dropout_rate=0.15):
        super(UNet3, self).__init__()
        self.best_model_val_loss = float("inf")
        self.best_model_val_acc = 0.0
        self.best_model_val_prec = 0.0
        self.best_model_val_rec = 0.0
        self.best_model_val_f1 = 0.0

        # Initialize metrics
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

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
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_classes)

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.binary_cross_entropy(y_pred, y)

        acc, prec, rec, f1 = self.compute_metrics(y_pred, y, stage="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_precision", prec, on_step=False, on_epoch=True)
        self.log("train_recall", rec, on_step=False, on_epoch=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.binary_cross_entropy(y_pred, y)

        acc, prec, rec, f1 = self.compute_metrics(y_pred, y, stage="val")
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)
        self.log("val_f1", f1)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        
        if self.best_model_val_loss < 0 or val_loss < self.best_model_val_loss:
            self.best_model_val_loss = val_loss
            self.best_model_val_acc = self.trainer.callback_metrics["val_acc"]
            self.best_model_val_prec = self.trainer.callback_metrics["val_precision"]
            self.best_model_val_rec = self.trainer.callback_metrics["val_recall"]
            self.best_model_val_f1 = self.trainer.callback_metrics["val_f1"]
        
        self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)
        self.log("best_model_val_acc", self.best_model_val_acc, prog_bar=True)
        self.log("best_model_val_prec", self.best_model_val_prec, prog_bar=True)
        self.log("best_model_val_rec", self.best_model_val_rec, prog_bar=True)
        self.log("best_model_val_f1", self.best_model_val_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_metrics(self, y_pred, y_true, stage="train"):
        y_pred_class = (y_pred > 0.5).float()
        if stage == "train":
            acc = self.train_accuracy(y_pred_class, y_true)
            prec = self.train_precision(y_pred_class, y_true)
            rec = self.train_recall(y_pred_class, y_true)
            f1 = self.train_f1(y_pred_class, y_true)
        else:
            acc = self.val_accuracy(y_pred_class, y_true)
            prec = self.val_precision(y_pred_class, y_true)
            rec = self.val_recall(y_pred_class, y_true)
            f1 = self.val_f1(y_pred_class, y_true)
        return acc, prec, rec, f1