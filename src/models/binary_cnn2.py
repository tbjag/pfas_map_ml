import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Initial convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),  # [N, 32, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [N, 32, H/2, W/2]
            nn.Dropout(0.3)
        )
        
        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_shortcut1 = nn.Conv2d(32, 64, kernel_size=1)
        
        # Intermediate block
        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(2),  # [N, 64, H/4, W/4]
            nn.Dropout(0.4)
        )
        
        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.res_shortcut2 = nn.Conv2d(64, 128, kernel_size=1)
        
        # Final processing
        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [N, 128, 1, 1]
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Initial block
        x = self.conv_block1(x)
        
        # Residual block 1
        residual = self.res_shortcut1(x)
        x = self.res_block1(x) + residual
        x = F.relu(x)
        
        # Intermediate processing
        x = self.conv_block2(x)
        
        # Residual block 2
        residual = self.res_shortcut2(x)
        x = self.res_block2(x) + residual
        x = F.relu(x)
        
        # Final classification
        x = self.final_block(x)
        return x