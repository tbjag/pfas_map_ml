import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

tensors = []
directory = 'data'
num_channels = 0
for filename in os.listdir(directory):
    tensor = torch.load(os.path.join(directory, filename))
    tensors.append(tensor)
    num_channels += 1

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Final layer to reduce channels to 18
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=18, kernel_size=3, padding=1)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutional layers and ReLU activations
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Output layer to reduce channels to 18
        x = self.output_conv(x)
        
        return x
    
model = Model()
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
num_epochs = 100
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

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig('losses2.png')

print("Training complete.")