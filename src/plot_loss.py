import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

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

        plots_dir = os.path.join(os.path.dirname(__file__), "..", "plots/testing", csv_folder, newest_version)
        plots_dir = os.path.abspath(plots_dir) 
        os.makedirs(plots_dir, exist_ok=True)

        # Plot training & validation loss with log scale
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", linestyle='-', marker='None')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", linestyle='-', marker='None')

        # Mark and label the lowest validation loss
        if val_loss:
            min_val = min(val_loss)
            min_epoch_idx = val_loss.index(min_val)
            min_epoch = min_epoch_idx + 1  # Epochs start at 1

            plt.scatter(min_epoch, min_val, color='red', zorder=5)
            plt.annotate(
                f'\nMin: {min_val:.4f}',
                xy=(min_epoch, min_val),
                xytext=(min_epoch + 0.5, min_val),
                ha='left',
                va='center',
                color='red',
            )

        # Add logarithmic scale and improve formatting
        plt.yscale('log')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (log scale)")
        plt.title(f"{plot_name}\nTraining and Validation Loss per Epoch")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().yaxis.set_major_formatter(LogFormatter())

        # Save the plot
        plot_name = "loss_plot.png"
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        return plots_dir
    else:
        return ""

if __name__ == "__main__":
    RUN_NAME = "iter3_all_10km_img_cnn4"
    csv_log_folder = RUN_NAME + "_csv"
    loss_plot_path = plot_loss(csv_folder=csv_log_folder, plot_name="CNN4 All 10km")