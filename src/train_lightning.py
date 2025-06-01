from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import dataloader as dt
from pytorch_lightning.utilities.model_summary import summarize
from models.CNNs.test_lightning import Model
from models.CNNs.test_lightning2 import Model as Model2
from models.CNNs.cnn4 import Model as CNN4
from models.UNets.unet3 import UNet3
from models.UNets.unet2 import UNet2
from models.UNets.unet1 import UNet1
from models.UNets.unet4 import UNet4
from models.UNets.unet1_3d import UNet1 as UNet1_3D
from models.CNNs.cnn2_3d import Model as Model2_3D
from plot_loss import plot_loss
from models.ResNets.resnet import ResNetModel as ResNet1
from models.ResNets.resnet2 import ResNetModel as ResNet2
from models.ResNets.resnet3 import ResNetModel as ResNet3
import os
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
torch.manual_seed(42)
np.random.seed(42)
import time
import pandas as pd

def get_metrics_for_best_checkpoint(csv_folder):
    logs_path = os.path.join(os.path.dirname(__file__), "..", "logs", csv_folder)
    logs_path = os.path.abspath(logs_path)

    version_folders = [f for f in os.listdir(logs_path) if f.startswith("version_")]
    if version_folders:
        newest_version = max(version_folders, key=lambda f: os.path.getmtime(os.path.join(logs_path, f)))
        csv_path = os.path.join(logs_path, newest_version, "metrics.csv")
        metrics_df = pd.read_csv(csv_path)

        # Filter to validation entries and group by epoch
        val_metrics = metrics_df.dropna(subset=['val_loss', 'val_r2', 'val_mae']).groupby('epoch').last()
        
        if not val_metrics.empty:
            best_row = val_metrics.loc[val_metrics['val_loss'].idxmin()]
            return {
                'best_r2': best_row['val_r2'],
                'best_mae': best_row['val_mae'],
                'best_epoch': best_row.name  # Returns the epoch number
            }
    
    return {'best_r2': -1, 'best_mae': -1, 'best_epoch': -1}

train_loader, test_loader = dt.get_dataloaders('/media/data/iter3/train/v4/all_10km', '/media/data/iter3/img_target/v1', 64, 8)

for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

targets = torch.cat([y for _, y in train_loader], dim=0)
print(f"Target range: [{targets.min()}, {targets.max()}]")
print(f"Target mean/std: {targets.mean():.4f}, {targets.std():.4f}")

RUN_NAME = "iter3_all_10km_pfas_unet2_decay=e-4"
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

# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=30,  # Number of epochs to wait before stopping
#     mode="min"
# )

# Train the model 
trainer = Trainer(
    gradient_clip_val=0.5,
    logger=[logger, csv_logger],
    callbacks=[checkpoint_callback],
    max_epochs=400
)
model = UNet2()
summary = summarize(model)

training_start_time = time.time()

trainer.fit(model, train_loader, test_loader)

training_duration = time.time() - training_start_time
hours = int(training_duration // 3600)
minutes = int((training_duration % 3600) // 60)
seconds = int(training_duration % 60)
training_time_str = f"{hours}h {minutes}m {seconds}s"

loss_plot_path = plot_loss(csv_folder=csv_log_folder, plot_name="UNet2 All 10km PFAS Weight Decay=e-4")
metrics = get_metrics_for_best_checkpoint(csv_log_folder)

# Save the stats for the model with the best validation loss to a text file
if loss_plot_path:
    print(loss_plot_path)
    stats_path = os.path.join(loss_plot_path, "best_model_stats.txt")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    print(stats_path)
    with open(stats_path, "w") as f:
        f.write(f"Best Model Validation Loss: {checkpoint_callback.best_model_score}\n")
        f.write(f"Best Model Validation R2: {metrics['best_r2']:.4f}\n")
        f.write(f"Best Model Validation MAE: {metrics['best_mae']:.4f}\n")
        f.write(f"Training Duration: {training_time_str}\n")
        f.write(f"Total Epochs: {trainer.current_epoch}\n")
        f.write(f"Min Loss at Epoch: {metrics['best_epoch']}\n")
        f.write(f"Best Model Path: {checkpoint_callback.best_model_path}\n")
    print("Matplotlib loss plot and best model stats text file saved to: " + loss_plot_path)
else:
    print("Failed plot and store stats")

print("Best model path:", checkpoint_callback.best_model_path)
print("Best validation loss:", checkpoint_callback.best_model_score)
print("R2 Score for best model:", metrics['best_r2'])
print("MAE Score for best model:", metrics['best_mae'])