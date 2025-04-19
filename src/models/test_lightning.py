import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.best_model_val_loss = -1
        
        # First convolutional layer
        # Input: 5x10x10, Output: 16x10x10
        # Padding='same' to maintain spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=36,
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
        
        # Final convolutional layer
        # Input: 8x10x10, Output: 1x10x10
        self.conv3 = nn.Conv2d(
            in_channels=8,
            out_channels=1,
            kernel_size=1  # 1x1 convolution for final channel reduction
        )
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Final 1x1 convolution
        x = self.conv3(x)
        return x

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
        # Get the aggregated validation loss for the epoch
        val_loss = self.trainer.callback_metrics["val_loss"]
        
        # Update best metrics if the current epoch's loss is better
        if self.best_model_val_loss < 0 or val_loss < self.best_model_val_loss:
            self.best_model_val_loss = val_loss
            
            # Log the best metrics
        self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


# import torch.nn as nn
# from pytorch_lightning import LightningModule
# from torch.optim import Adam
# import torchmetrics

# class Model(LightningModule):
#     def __init__(self):
#         super(Model, self).__init__()
#         # Regression metrics
#         self.train_mse = torchmetrics.MeanSquaredError()
#         self.train_mae = torchmetrics.MeanAbsoluteError()
        
#         self.val_mse = torchmetrics.MeanSquaredError()
#         self.val_mae = torchmetrics.MeanAbsoluteError()

#         # Best model tracking - same pattern as classification
#         self.best_model_val_loss = float("inf")
#         self.best_model_val_mse = float("inf")
#         self.best_model_val_mae = float("inf")

#         # Network architecture (your existing layers)
#         self.conv1 = nn.Conv2d(32, 16, 3, padding='same')
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(16, 8, 3, padding='same')
#         self.bn2 = nn.BatchNorm2d(8)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv2d(8, 1, 1)

#     def forward(self, x):
#         # Your existing forward pass
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         return self.conv3(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self(x)
#         loss = nn.functional.mse_loss(y_pred, y)
#         mse, mae = self.compute_metrics(y_pred, y, stage="train")

#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("train_mse", mse, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("train_mae", mae, on_step=False, on_epoch=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self(x)
#         loss = nn.functional.mse_loss(y_pred, y)
#         mse, mae = self.compute_metrics(y_pred, y, stage="val")

#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_mse", mse)
#         self.log("val_mae", mae)
#         return loss
    
#     def on_validation_epoch_end(self):
#         # Get the aggregated validation metrics
#         val_loss = self.trainer.callback_metrics["val_loss"]
        
#         # Update best metrics if current epoch is better
#         if val_loss < self.best_model_val_loss:
#             self.best_model_val_loss = val_loss
#             self.best_model_val_mse = self.trainer.callback_metrics["val_mse"]
#             self.best_model_val_mae = self.trainer.callback_metrics["val_mae"]
            
#             # Log the best metrics with consistent naming
#             self.log("best_model_val_loss", self.best_model_val_loss, prog_bar=True)
#             self.log("best_model_val_mse", self.best_model_val_mse, prog_bar=True)
#             self.log("best_model_val_mae", self.best_model_val_mae, prog_bar=True)

#     def configure_optimizers(self):
#         return Adam(self.parameters(), lr=1e-3)
    
#     def compute_metrics(self, y_pred, y_true, stage="train"):
#         # Flatten spatial dimensions for metrics
#         y_pred_flat = y_pred.flatten(start_dim=1)  # [batch, 32*32]
#         y_true_flat = y_true.flatten(start_dim=1)

#         if stage == "train":
#             mse = self.train_mse(y_pred_flat, y_true_flat)
#             mae = self.train_mae(y_pred_flat, y_true_flat)
#         else:  # validation
#             mse = self.val_mse(y_pred_flat, y_true_flat)
#             mae = self.val_mae(y_pred_flat, y_true_flat)

#         return mse, mae