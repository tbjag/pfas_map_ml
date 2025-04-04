import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
import torchmetrics

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        
        # First convolutional layer
        # Input: 5x10x10, Output: 16x10x10
        # Padding='same' to maintain spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=263,
            out_channels=16,
            kernel_size=3,
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # Second convolutional layer
        # Input: 16x10x10, Output: 8x10x10
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        
        # Global average pooling to collapse spatial dimensions (32x32 → 1x1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(8, 1)  # 8 channels → 1 output
        
        # Sigmoid activation for probability
        self.sigmoid = nn.Sigmoid()

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")
        self.train_f1 = torchmetrics.F1Score(task="binary")

        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Global pooling and reshape
        x = self.global_pool(x)  # Shape: [batch, 8, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch, 8]
        
        # Fully connected + sigmoid
        x = self.fc(x)
        x = self.sigmoid(x)
        return x  # Output shape: [batch, 1]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.binary_cross_entropy(y_pred, y)
        acc, prec, rec, f1 = self.compute_metrics(y_pred, y, stage="train")

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_precision", prec)
        self.log("train_recall", rec)
        self.log("train_f1", f1)
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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def compute_metrics(self, y_pred, y_true, stage="train"):
        y_pred_class = (y_pred > 0.5).float()  # Convert probabilities to binary values (0 or 1)

        if stage == "train":
            acc = self.train_accuracy(y_pred_class, y_true)
            prec = self.train_precision(y_pred_class, y_true)
            rec = self.train_recall(y_pred_class, y_true)
            f1 = self.train_f1(y_pred_class, y_true)
        else:  # validation
            acc = self.val_accuracy(y_pred_class, y_true)
            prec = self.val_precision(y_pred_class, y_true)
            rec = self.val_recall(y_pred_class, y_true)
            f1 = self.val_f1(y_pred_class, y_true)

        return acc, prec, rec, f1
