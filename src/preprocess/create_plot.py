import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Set input and output directories
input_folder = '/media/data/iter3/raw_raster/roads'
output_folder = '/media/data/iter3/plots'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all .tif files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.tif'):
        raster_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")

        print(f"Processing: {raster_path}")

        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            raster_data = np.ma.masked_equal(raster_data, src.nodata)

            # Plot setup
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(raster_data, cmap='inferno')
            fig.colorbar(cax, ax=ax)
            ax.set_title(f'Raster Plot: {filename}')
            ax.axis('off')

            # Save figure
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)  # Close the figure to free memory

            print(f"Saved plot to: {output_path}")
