### Prints the newest loss plot of the lightning model

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(csv_folder="iter3_binary_csv", plot_name=""):

    logs_path = os.path.join(os.path.dirname(__file__), "..", "logs", csv_folder)
    logs_path = os.path.abspath(logs_path)  # Convert to absolute path


    # Get all version folders
    version_folders = [f for f in os.listdir(logs_path) if f.startswith("version_")]

    if version_folders:
        newest_version = max(version_folders, key=lambda f: os.path.getmtime(os.path.join(logs_path, f)))
        metrics_path = os.path.join(logs_path, newest_version, "metrics.csv")

        # Read CSV file
        df = pd.read_csv(metrics_path)

        # Drop rows where all values are NaN
        df.dropna(how="all", inplace=True)

        # Extract val_loss and train_loss into separate lists
        val_loss = df["val_loss"].dropna().tolist()
        train_loss = df["train_loss"].dropna().tolist()

        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots", csv_folder, newest_version)  # Get 'plots' folder
        plots_dir = os.path.abspath(plots_dir) 
        os.makedirs(plots_dir, exist_ok=True)

        # Extract loss values
        val_loss = df["val_loss"].dropna().tolist()
        train_loss = df["train_loss"].dropna().tolist()

        # Plot training & validation loss with log scale
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", linestyle='-', marker='None')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", linestyle='-', marker='None')

        # Add logarithmic scale and improve formatting
        plt.yscale('log')  # <--- KEY ADDITION
        plt.xlabel("Epochs")
        plt.ylabel("Loss (log scale)")  # Update label
        plt.title(f"{plot_name}\nTraining and Validation Loss per Epoch")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Show both major/minor grid

        # Optional: Set custom ticks for better readability
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
        plt.gca().yaxis.set_minor_formatter(plt.ScalarFormatter())

        # Save the plot
        plot_name = "loss_plot_log.png"  # Different name for log version
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        return plots_dir
    else:
        return ""