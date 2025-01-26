import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        
        # First convolutional layer
        # Input: 5x10x10, Output: 16x10x10
        # Padding='same' to maintain spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=4,
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
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    
