import dataloader as dt
from models.test_cnn import Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

model = Model()
criterion = nn.MSELoss()  # Adjust this if using a different loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = dt.get_dataloaders('../data/final_train', '../data/final_target', 32)

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
        
        total_loss += loss.item()  # * inputs.size(0)  # Accumulate batch loss

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
            total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 100  # Set the number of epochs
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

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(test_losses, label='Test Loss', color='orange', marker='o')

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()

# Save the figure
plt.savefig('loss_plot.png', dpi=300)

print("Training complete.")