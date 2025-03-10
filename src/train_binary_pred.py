from models.binary_cnn import Model
import torch
import matplotlib.pyplot as plt
import json
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pandas as pd

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
        input_tensor = torch.load(input_path)
        #print(f"File: {file_name}, Shape: {input_tensor.shape}")
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

binary_target = os.path.join(os.path.dirname(__file__), "/media/data/iter3/bin_target/", "binary_target.json")
train_loader, test_loader = get_dataloaders('/media/data/iter3/train', binary_target, 8, 1)


for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

# Set up logging and checkpointing
logger = TensorBoardLogger("logs", name="iter3_binary")
csv_logger = CSVLogger("logs", name="iter3_binary")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    save_top_k=1,        # Save only the best model
    mode="min"           # Minimize the monitored metric (e.g., val_loss)
)

# Train the model
trainer = Trainer(
    logger=[logger, csv_logger],
    callbacks=[checkpoint_callback],
    max_epochs=2
)
model = Model()
trainer.fit(model, train_loader, test_loader)

# Retrieve logged metrics from the trainer
metrics = trainer.logged_metrics  # Access metrics logged during training

best_val_loss = trainer.logged_metrics["best_model_val_loss"]
best_val_acc = trainer.logged_metrics["best_model_val_acc"]
print(f"Best model validation loss: {best_val_loss}")
print(f"Best model validation accuracy: {best_val_acc}")

# Extract loss values
# train_losses = trainer.callback_metrics.get("train_loss").cpu().numpy()
# val_losses = trainer.callback_metrics.get("val_loss").cpu().numpy()

# Plot the losses
# plt.plot(train_losses, label="Training Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training & Validation Loss Over Time")
# plt.legend()
# plt.savefig("plot.png")
# plt.show()

print("Best model path:", checkpoint_callback.best_model_path)
print("Best validation loss:", checkpoint_callback.best_model_score)