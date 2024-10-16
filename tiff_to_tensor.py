import rasterio
import torch
import geopandas as gpd
from rasterio.mask import mask
import os
import numpy as np

def tiff_to_tensor(tiff_file, grid_size, output_directory):
    # Load the TIFF file
    with rasterio.open(tiff_file) as src:
        tiff_data = src.read()

        original_bands, original_height, original_width = tiff_data.shape
        print(f'Original image dimensions: {original_bands} bands, {original_height} height, {original_width} width')

        cells = []

        for i in range(0, original_height, grid_size):
            for j in range(0, original_width, grid_size):
                cell = tiff_data[:, i:i+grid_size, j:j+grid_size]
                if cell.shape == (1, grid_size, grid_size):
                    # Check if the cell contains any null values (assuming null values are represented by NaNs)
                    if not np.any(np.isnan(cell)) and not np.any(cell < 0):
                        cells.append(cell)
 

    if cells:
        tensors = [torch.tensor(cell, dtype=torch.float32) for cell in cells]
        stacked_tensor = torch.stack(tensors)
        print(f'Stacked tensor dimensions: {stacked_tensor.shape}')

        base_name = os.path.splitext(os.path.basename(tiff_file))[0]
        output_file = os.path.join(output_directory, f'{base_name}.pth')
        torch.save(stacked_tensor, output_file)

        print(f'Tensor saved to {output_file}')
    else:
        print('No valid cells found, tensor not saved.')


#Create Stacked Tensors for each file in selected directory
def process_directory(directory, grid_size, output_directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                tiff_file = os.path.join(root, file)
                try:
                    tiff_to_tensor(tiff_file, grid_size, output_directory)
                except Exception as e:
                    print(f"Failed to process {tiff_file}: {e}")

def main():
    input_dir = '.'
    grid_size = 10
    output_dir = 'out_pth'
    
    # process_directory(input_dir, grid_size, output_dir)

    tiff_to_tensor('output_doubled.tiff', grid_size, 'out_pth')

if __name__ == "__main__":
    main()