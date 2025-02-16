import torch.nn as nn

class SimplestCNN(nn.Module):
    def __init__(self):
        super(SimplestCNN, self).__init__()
        
        # First convolutional layer
        # Input: 263x32x32, Output: 8x32x32
        self.conv1 = nn.Conv2d(
            in_channels=263,  
            out_channels=8,   
            kernel_size=3,    
            padding=1  
        )
        self.bn1 = nn.BatchNorm2d(8)  # Batch Norm for first layer
        self.relu1 = nn.ReLU()

        # Final convolutional layer (reducing to 1 channel)
        # Input: 8x32x32, Output: 1x32x32
        self.conv2 = nn.Conv2d(
            in_channels=8, 
            out_channels=1, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu1(x)  

        x = self.conv2(x)

        return x