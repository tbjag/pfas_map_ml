import dataloader as dt
from models.binary_cnn import Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, label_json_path):
        self.input_dir = input_dir
        self.input_files = os.listdir(input_dir)  # List of .pth files
        # Load JSON labels
        with open(label_json_path, 'r') as f:
            self.labels = json.load(f)
        self.input_files = [f for f in self.input_files if f in self.labels]
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # Load input tensor
        file_name = self.input_files[idx]
        input_path = os.path.join(self.input_dir, file_name)
        input_tensor = torch.load(input_path)  # Shape: [263, 32, 32]
        # Get label from JSON (ensure key matches filename)
        label = self.labels[file_name]
        label_tensor = torch.tensor(label, dtype=torch.float32).view(1)  # BCELoss needs float
        return input_tensor, label_tensor

def get_dataloaders(input_dir, label_json_path, batch_size, num_workers=1):
    # Create dataset using input_dir (folder of .pth files) and label JSON
    full_dataset = BinaryDataset(input_dir, label_json_path)
    
    # Split into train/test (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

binary_target = os.path.join(os.path.dirname(__file__), "binary_classification", "binary_target.json")
train_loader, test_loader = get_dataloaders('/media/data/iter2/train', binary_target, 8, 1)


for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

# Training and evaluation functions
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, target in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        target = target.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, target)  # Assuming a reconstruction task
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
        for inputs, target in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            total_loss += loss.item()  * inputs.size(0)  # Accumulate batch loss

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

model = Model()
criterion = nn.BCELoss()  # Needed for binary classification
optimizer = optim.Adam(model.parameters(), 1e-4) # try lr=0.001 if performance is poor

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 5  # Set the number of epochs
train_losses = []
test_losses = []

min_test_loss = -1
min_epoch = -1

training_name = "dummy"

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

    if min_test_loss == -1 or test_loss < min_test_loss:
        min_test_loss = test_loss
        min_epoch = epoch
        torch.save(model.state_dict(), f"trained_models/{training_name}.pth")

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(test_losses, label='Test Loss', color='orange', marker='o')

plt.scatter(min_epoch, min_test_loss, color='red', label=f'Min Test Loss: {min_test_loss:.2f} (Epoch {min_epoch+1})')

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{training_name}: Training and Test Loss Over Epochs')
plt.legend()

# Save the figure
plt.savefig(f'plots/{training_name}.png', dpi=300)

print("Training complete.")