from models.CNNs.cnn_binary import Model
from models.CNNs.cnn2_binary import Model as Model2
from models.CNNs.cnn3_binary import Model as Model3
import torch
import matplotlib.pyplot as plt
import json
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pandas as pd
from plot_loss import plot_loss
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
torch.set_float32_matmul_precision("high")
from pytorch_lightning.utilities.model_summary import summarize
from models.UNets.unet1_binary import UNet1
from models.UNets.unet2_binary import UNet2
from models.UNets.unet3_binary import UNet3

def get_best_classification_metrics(csv_folder):
    """Returns dict of best validation metrics (loss, acc, prec, rec, f1)"""
    logs_path = os.path.join(os.path.dirname(__file__), "..", "logs", csv_folder)
    logs_path = os.path.abspath(logs_path)

    version_folders = [f for f in os.listdir(logs_path) if f.startswith("version_")]
    if not version_folders:
        return None  # or raise error

    newest_version = max(version_folders, key=lambda f: os.path.getmtime(os.path.join(logs_path, f)))
    csv_path = os.path.join(logs_path, newest_version, "metrics.csv")
    metrics_df = pd.read_csv(csv_path)

    # Get row with best validation loss (or any other metric you monitor)
    best_epoch_row = metrics_df.loc[metrics_df["val_loss"].idxmin()]

    return {
        "val_acc": best_epoch_row["val_acc"],
        "val_prec": best_epoch_row["val_prec"],
        "val_rec": best_epoch_row["val_rec"],
        "val_f1": best_epoch_row["val_f1"]
    }

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
train_loader, test_loader = get_dataloaders('/media/data/iter3/train/v1/avg_temp+csvs', binary_target, 64, 8)


for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

RUN_NAME = "iter3_avg_temp+csvs_binary_2"
tensorboard_log_folder = RUN_NAME + "_tensorboard"
csv_log_folder = RUN_NAME + "_csv"

# Set up logging and checkpointing
logger = TensorBoardLogger("logs", name=tensorboard_log_folder)
csv_logger = CSVLogger("logs", name=csv_log_folder)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    save_top_k=1,
    mode="min"
)

# Train the model
trainer = Trainer(
    logger=[logger, csv_logger],
    callbacks=[checkpoint_callback],
    max_epochs=10
)
model = Model2()

summary = summarize(model)

trainer.fit(model, train_loader, test_loader)

loss_plot_path = plot_loss(csv_folder=csv_log_folder, plot_name="CNN2 avg_temp+csvs Logistic")

metrics = get_best_classification_metrics(csv_log_folder)

# Save the stats for the model with the best validation loss to a text file
best_val_loss = checkpoint_callback.best_model_score
best_val_acc = metrics["val_acc"]
best_val_rec = metrics["val_rec"]
best_val_prec = metrics["val_prec"]
best_val_f1 = metrics["val_f1"]
if loss_plot_path:
    print(loss_plot_path)
    stats_path = os.path.join(loss_plot_path, "best_model_stats.txt")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    
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
print("Accuracy for best model", best_val_acc)
print("Recall for best model", best_val_rec)
print("Precision for best model", best_val_prec)
print("F1 for best model", best_val_f1)