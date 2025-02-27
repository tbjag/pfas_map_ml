import dataloader as dt
from models.test_cnn import Model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.unet1 import UNet
from models.unet2 import UNet2  
from models.unet3 import UNet3
import models.test_cnn as test_cnn

from train import train_one_epoch, evaluate

model = UNet3(n_channels=263, n_classes=1)
#model = EnhancedModel()
# model = test_cnn.Model()
criterion = nn.MSELoss()  # Adjust this if using a different loss
optimizer = optim.Adam(model.parameters(), lr=1e-4) # weight_decay=1e-5

train_loader, test_loader = dt.get_dataloaders('/media/data/iter2/train', '/media/data/iter2/target', 8)

for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 250  # Set the number of epochs
train_losses = []
test_losses = []

min_test_loss = -1
min_epoch = -1

# Load the saved model and continue training
checkpoint_path = "trained_models/unet3_25dr_500e.pth"

# Load the model weights
if torch.cuda.is_available():
    model.load_state_dict(torch.load(checkpoint_path))  # Load to GPU if available
else:
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))  # Load to CPU

print(f"Loaded model from {checkpoint_path}")

# Move the model to the device
model.to(device)

# Resume optimizer state if required (not mandatory but recommended for consistency in optimizer behavior)
# optimizer.load_state_dict(torch.load("path_to_optimizer_state.pth"))  # Uncomment if you saved optimizer state earlier

# Continue training
additional_epochs = 250  # Number of additional epochs to train
for epoch in range(num_epochs, num_epochs + additional_epochs):
    print(f"Epoch {epoch+1}/{num_epochs + additional_epochs}")
    
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
        torch.save(model.state_dict(), checkpoint_path)  # Save model after improvement
        print(f"Model improved. Saved to {checkpoint_path}")

# Plot the updated losses
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

# Save the updated figure
plt.savefig('plots/unet3_25dr_extended_training.png', dpi=300)

print("Extended training complete.")
