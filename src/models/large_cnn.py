import torch
import torch.nn as nn

class EfficientResidualBlock(nn.Module):
    def __init__(self, channels):
        super(EfficientResidualBlock, self).__init__()
        # Using groups in conv layers to reduce parameters
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)  # inplace operation to save memory
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Initial feature extraction with fewer channels
        self.conv1 = nn.Conv2d(263, 32, kernel_size=3, padding=1, groups=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Moderate size processing blocks
        self.layer1 = self._make_layer(32, 48)
        self.layer2 = self._make_layer(48, 64)
        self.layer3 = self._make_layer(64, 48)
        self.layer4 = self._make_layer(48, 32)
        
        # Efficient final layers
        self.final_conv1 = nn.Conv2d(32, 16, kernel_size=1)
        self.final_bn1 = nn.BatchNorm2d(16)
        self.final_conv2 = nn.Conv2d(16, 1, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(EfficientResidualBlock(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Feature processing
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final prediction
        x = self.final_conv1(x)
        x = self.final_bn1(x)
        x = self.relu(x)
        x = self.final_conv2(x)
        
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())