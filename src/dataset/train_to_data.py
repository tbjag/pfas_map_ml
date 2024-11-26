# calculate how many single data entities there are

# create empty 0xSIZExSIZE for train and target = (target is SIZExSIZE only)

# go through each tiff and populate each file with extra dim

# in dataloader load by file and concat/add

import rasterio
import torch
import os
import numpy as np

def tiff_to_tensor(tiff_file, grid_size, output_directory, first):
    # Load the TIFF file
    cnt = 0
    with rasterio.open(tiff_file) as src:
        tiff_data = src.read()

        _, original_height, original_width = tiff_data.shape

        for i in range(0, original_height, grid_size):
            for j in range(0, original_width, grid_size):
                cell = tiff_data[:, i:i+grid_size, j:j+grid_size]
                if cell.shape == (1, grid_size, grid_size):
                    if not np.any(np.isnan(cell)) or not np.any(cell < 0):
                        filepath = os.path.join(output_directory, f'{cnt:07}.pth')
                        cell -= 1
                        out_tensor = torch.tensor(cell, dtype=torch.float32)
                        if not first:
                            comb = torch.load(filepath)
                            out_tensor = torch.cat((comb, out_tensor), dim=0)
                        torch.save(out_tensor, filepath)
                        cnt += 1

#Create Stacked Tensors for each file in selected directory
def process_directory(directory, grid_size, output_directory):
    first = True
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tiff'):
                tiff_file = os.path.join(root, file)
                try:
                    tiff_to_tensor(tiff_file, grid_size, output_directory, first)
                    if first:
                        first = False
                except Exception as e:
                    print(f"Failed to process {tiff_file}: {e}")
    print('finished')
    
def main():
    input_dir = '../../data/tiff_temp'
    grid_size = 32
    output_dir = '../../data/final_train'
    
    process_directory(input_dir, grid_size, output_dir)


if __name__ == "__main__":
    main()