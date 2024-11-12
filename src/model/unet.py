import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from tqdm import tqdm

tensors = []
directory = 'pth_data_small'
for filename in os.listdir(directory):
    tensor = torch.load(os.path.join(directory, filename))
    tensors.append(tensor)

input_tensor = torch.cat(tensors, dim=1)

target_tensor = torch.load('HUC8_CA_PFAS_GTruth_Summa2.pth')

print('input tensor shape:', input_tensor.shape, 'target shape: ', target_tensor.shape)
concat_tensor = TensorDataset(input_tensor, target_tensor)

# Define split sizes for training and test sets (e.g., 80-20 split)
train_size = int(0.8 * len(concat_tensor))
test_size = len(concat_tensor) - train_size
train_data, test_data = random_split(concat_tensor, [train_size, test_size])

# Wrap in DataLoader for batch processing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Reduce to a single channel output while keeping the spatial dimensions
        self.output_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # Input x has shape [batch_size, 18, 10, 10]
        
        x = F.relu(self.conv1(x))  # Shape: [batch_size, 32, 10, 10]
        x = F.relu(self.conv2(x))  # Shape: [batch_size, 64, 10, 10]
        x = F.relu(self.conv3(x))  # Shape: [batch_size, 128, 10, 10]
        
        # Final layer to get a single output channel (10x10x1)
        x = self.output_conv(x)    # Shape: [batch_size, 1, 10, 10]
        
        return x  # Output shape: [batch_size, 1, 10, 10]

    
model = CNNModel()
criterion = nn.MSELoss()  # Adjust this if using a different loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation functions
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        inputs = batch[0].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Assuming a reconstruction task
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)  # Accumulate batch loss

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 2  # Set the number of epochs
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    print(f"Training Loss: {train_loss:.4f}")

    # Evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    print(f"Test Loss: {test_loss:.4f}")

print("Training complete.")