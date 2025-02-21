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
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.binary_cross_entropy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
