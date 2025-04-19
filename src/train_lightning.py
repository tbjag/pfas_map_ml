from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import dataloader as dt
from pytorch_lightning.utilities.model_summary import summarize
from models.test_lightning import Model
from plot_loss import plot_loss
import os

train_loader, test_loader = dt.get_dataloaders('/media/data/iter3/train/avg_temp+csvs', '/media/data/iter3/img_target', 64, 8)

for inputs, target in train_loader:
    print("Training Data Shape:", inputs.shape)
    print("Training Target Shape:", target.shape)
    break  # Print shape for only the first batch

RUN_NAME = "iter3_avg_temp+csvs_img"
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
    max_epochs=500
)
model = Model()
summary = summarize(model)
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
    print(stats_path)
    with open(stats_path, "w") as f:
        f.write(f"Best Model Validation Loss: {best_val_loss}\n")
    print("Matplotlib loss plot and best model stats text file saved to: " + loss_plot_path)
else:
    print("Failed plot and store stats")

print("Best model path:", checkpoint_callback.best_model_path)
print("Best validation loss:", checkpoint_callback.best_model_score)