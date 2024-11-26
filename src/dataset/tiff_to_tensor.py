import rasterio
import torch
import geopandas as gpd
from rasterio.mask import mask
import os
import numpy as np

def tiff_to_tensor(tiff_file, grid_size, output_directory, batch_size):
    cnt = 1
    batch_cnt = 0

    # Load the TIFF file
    with rasterio.open(tiff_file) as src:
        tiff_data = src.read()

        _, original_height, original_width = tiff_data.shape
        cells = []

        for i in range(0, original_height, grid_size):
            for j in range(0, original_width, grid_size):
                cell = tiff_data[:, i:i+grid_size, j:j+grid_size]
                if cell.shape == (1, grid_size, grid_size):
                    # Check if the cell contains any null values (assuming null values are represented by NaNs)
                    if not np.any(np.isnan(cell)) or not np.any(cell < 0):
                        cell -= 1
                        cells.append(cell)

                        if cnt == batch_size:
                            tensors = [torch.tensor(cell, dtype=torch.float32) for cell in cells]
                            stacked_tensor = torch.stack(tensors)
                            output_file = os.path.join(output_directory, f'{batch_cnt:03}.pth')
                            torch.save(stacked_tensor, output_file)
                            batch_cnt += 1
                            cells = []
                            cnt = 0

                        cnt += 1

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
                base_name = os.path.splitext(os.path.basename(tiff_file))[0]
                out_dir = os.path.join(output_directory, base_name)
                os.mkdir(out_dir)
                print(out_dir)
                try:
                    tiff_to_tensor(tiff_file, grid_size, out_dir, 1)
                except Exception as e:
                    print(f"Failed to process {tiff_file}: {e}")
    print('finished')
    
def main():
    input_dir = '../../data/tiff_temp'
    grid_size = 32
    output_dir = '../../data/large_files_train'
    
    process_directory(input_dir, grid_size, output_dir)


if __name__ == "__main__":
    main()