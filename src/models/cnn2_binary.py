import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim import Adam

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")
        
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

        self.best_model_val_loss = float("inf")
        self.best_model_val_acc = 0.0
        self.best_model_val_prec = 0.0
        self.best_model_val_rec = 0.0
        self.best_model_val_f1 = 0.0

        # Convolutional layers
        self.conv1 = nn.Conv2d(50, 256, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.2)

        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(8)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout5(self.relu5(self.bn5(self.conv5(x))))
        x = self.dropout6(self.relu6(self.bn6(self.conv6(x))))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
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
        return loss

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        if val_loss < self.best_model_val_loss:
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
        return Adam(self.parameters(), lr=1e-3)

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
