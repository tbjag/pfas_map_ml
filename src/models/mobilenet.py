import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class ModifiedMobileNetV3(nn.Module):
    def __init__(self, input_channels=263, output_channels=1):
        super(ModifiedMobileNetV3, self).__init__()
        # Load pre-trained MobileNetV3-Small
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # Modify the first convolutional layer to accept 263 input channels
        original_first_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )
        
        # Modify the classifier head for output size of 1
        self.mobilenet.classifier[3] = nn.Linear(1024, output_channels)

    def forward(self, x):
        return self.mobilenet(x)
    

# # Load MobileNetV2 (or V3)
# model = models.mobilenet_v2(pretrained=True)  # You can also use mobilenet_v3_large()

# # Modify the first convolutional layer if your data has 263 channels
# # MobileNet expects 3-channel input, so we replace the first layer
# model.features[0][0] = nn.Conv2d(
#     in_channels=263,  # Your input channels
#     out_channels=32,  # Keep MobileNet's default out_channels
#     kernel_size=3,
#     stride=2,
#     padding=1,
#     bias=False
# )

# # Modify the classifier to output the correct shape (1x32x32)
# model.classifier = nn.Sequential(
#     nn.Linear(model.last_channel, 1024),  # Intermediate FC layer
#     nn.ReLU(),
#     nn.Linear(1024, 32 * 32),  # Output size adjusted
#     nn.Unflatten(1, (1, 32, 32))  # Reshape to match target shape
# )