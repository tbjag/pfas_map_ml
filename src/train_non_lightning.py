import torch
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import dataloader as dt
from models.CNNs.test_lightning2 import Model
from models.CNNs.cnn2_3d import Model as Model_3D

torch.manual_seed(42)
np.random.seed(42)

train_loader, test_loader = dt.get_dataloaders('/media/data/iter3/train/v4/all_10km', '/media/data/iter3/img_target/v1', 64, 8)


for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

model = Model()  # Replace with your model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, target in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        target = target.to(device).float()

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()  # Accumulate batch loss

    avg_loss = total_loss / len(loader)  # Average loss per batch
    return avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, target in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            target = target.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)  # Average loss per batch
    return avg_loss

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 100  # Set the number of epochs
train_losses = []
test_losses = []

min_test_loss = float('inf')  # Track the minimum test loss
min_epoch = -1  # Track the epoch with the minimum test loss

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

    # Save the model if test loss improves
    if test_loss < min_test_loss:
        min_test_loss = test_loss
        min_epoch = epoch
        torch.save(model.state_dict(), "trained_models/test_lightning2.pth")
        print(f"New best model saved with Test Loss: {min_test_loss:.4f}")

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(test_losses, label='Test Loss', color='orange', marker='o')

# Highlight the minimum test loss point
plt.scatter(min_epoch, min_test_loss, color='red', label=f'Min Test Loss: {min_test_loss:.2f} (Epoch {min_epoch+1})')

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()

# Save the figure
plt.savefig('loss_plot_iter3_img.png', dpi=300)

print("Training complete.")
print(f'Min Test Loss: {min_test_loss:.2f}')