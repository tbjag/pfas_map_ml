from models.binary_cnn import Model
import torch
import matplotlib.pyplot as plt
import json
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pandas as pd
from plot_loss import plot_loss

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

RUN_NAME = "iter3_binary"
tensorboard_log_folder = RUN_NAME + "_tensorboard"
csv_log_folder = RUN_NAME + "_csv"

# Set up logging and checkpointing
logger = TensorBoardLogger("logs", name=tensorboard_log_folder)
csv_logger = CSVLogger("logs", name=csv_log_folder)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    save_top_k=1,        # Save only the best model
    mode="min"           # Minimize the monitored metric (e.g., val_loss)
)

# Train the model
trainer = Trainer(
    logger=[logger, csv_logger],
    callbacks=[checkpoint_callback],
    max_epochs=3
)
model = Model()
trainer.fit(model, train_loader, test_loader)

# make matplot lib loss plot
loss_plot_path = plot_loss(csv_log_folder)

# Retrieve logged metrics from the trainer
metrics = trainer.logged_metrics  # Access metrics logged during training

# Save the stats for the model with the best validation loss to a text file
if loss_plot_path:
    print(loss_plot_path)
    stats_path = os.path.join(loss_plot_path, "best_model_stats.txt")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    best_val_loss = trainer.logged_metrics["best_model_val_loss"]
    best_val_acc = trainer.logged_metrics["best_model_val_acc"]
    best_val_rec = trainer.logged_metrics["best_model_val_rec"]
    best_val_prec = trainer.logged_metrics["best_model_val_prec"]
    best_val_f1 = trainer.logged_metrics["best_model_val_f1"]
    print(stats_path)
    with open(stats_path, "w") as f:
        f.write(f"Best Model Validation Loss: {best_val_loss}\n")
        f.write(f"Best Model Validation Accuracy: {best_val_acc}\n")
        f.write(f"Best Model Validation Recall: {best_val_rec}\n")
        f.write(f"Best Model Validation Precision: {best_val_prec}\n")
        f.write(f"Best Model Validation F1 Score: {best_val_f1}\n")
    print("Matplotlib loss plot and best model stats text file saved to: " + loss_plot_path)
else:
    print("Failed plot and store stats")

print("Best model path:", checkpoint_callback.best_model_path)
print("Best validation loss:", checkpoint_callback.best_model_score)