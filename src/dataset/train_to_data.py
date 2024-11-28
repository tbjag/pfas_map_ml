import rasterio
import torch
import os
import numpy as np
import argparse
from tqdm import tqdm

def tiff_to_tensor(tiff_file, grid_size, output_directory, first=True):
    try:
        cnt = 0
        with rasterio.open(tiff_file) as src:
            tiff_data = src.read()

            # Handle potential multi-band or single-band TIFF
            if len(tiff_data.shape) == 2:
                tiff_data = tiff_data[np.newaxis, :, :]
            
            _, original_height, original_width = tiff_data.shape

            for i in range(0, original_height - grid_size + 1, grid_size):
                for j in range(0, original_width - grid_size + 1, grid_size):
                    cell = tiff_data[:, i:i+grid_size, j:j+grid_size]
                    
                    # Validate cell dimensions and data
                    if cell.shape[0] > 0 and cell.shape[1] == grid_size and cell.shape[2] == grid_size:
                        # Check for valid data (no NaNs and no negative values)
                        if not np.isnan(cell).any() and not (cell < 0).any():
                            filepath = os.path.join(output_directory, f'{cnt:07d}.pth')
                            
                            # Subtract 1 from cell values
                            cell = cell.astype(np.float32) - 1
                            
                            out_tensor = torch.tensor(cell, dtype=torch.float32)
                            
                            if not first and os.path.exists(filepath):
                                comb = torch.load(filepath)
                                out_tensor = torch.cat((comb, out_tensor), dim=0)
                            
                            torch.save(out_tensor, filepath)
                            cnt += 1
        
        return cnt
    except Exception as e:
        print(f"Error processing {tiff_file}: {e}")
        return 0

def process_directory(files, grid_size, output_directory):
    total_entities = 0
    for file in tqdm(files):
        try:
            file_entities = tiff_to_tensor(file, grid_size, output_directory)
            total_entities += file_entities
        except Exception as e:
            print(f"Failed to process {file}: {e}")
    
    print(f'Total single data entities processed: {total_entities}')

def check_paths(dir_path, output_dir):
    if not os.path.isdir(dir_path):
        raise ValueError(f'Input folder {dir_path} does not exist')
    
    if not os.path.isdir(output_dir):
        raise ValueError(f'Output folder {output_dir} does not exist')

def queue_files(dir_path):
    tiff_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.tiff', '.tif')):
                full_path = os.path.join(root, file)
                tiff_files.append(full_path)
    
    if not tiff_files:
        raise ValueError(f'No TIFF files found in {dir_path}')
    
    return tiff_files

def main():
    parser = argparse.ArgumentParser(description='Process TIFF files into tensors')
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory path")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory path")
    parser.add_argument('--grid_size', type=int, required=True, help="Grid size for processing")

    args = parser.parse_args()
    
    try:
        # Validate paths
        check_paths(args.input_dir, args.output_dir)

        # Find TIFF files
        files = queue_files(args.input_dir)

        # Process files
        process_directory(files, args.grid_size, args.output_dir)

        print('Processing completed successfully')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()