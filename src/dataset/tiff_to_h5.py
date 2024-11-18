import rasterio
import h5py
import os
import numpy as np
import argparse
from tqdm import tqdm

def tiff_to_tensor(tiff_file, grid_size, output_directory):
    # Load the TIFF file
    with rasterio.open(tiff_file) as src:
        tiff_data = src.read()

        _, original_height, original_width = tiff_data.shape
        cells = []

        for i in tqdm(range(0, original_height, grid_size), desc=f'processing {tiff_file}'):
            for j in range(0, original_width, grid_size):
                cell = tiff_data[:, i:i+grid_size, j:j+grid_size]
                if cell.shape == (1, grid_size, grid_size): # maybe just save as 10x10 instead of 1x10x10 TODO
                    # Check if the cell contains any null values (assuming null values are represented by NaNs)
                    if not np.any(np.isnan(cell)) or not np.any(cell < 0):
                        cell -= 1
                        cells.append(cell)
    print(f'saving {len(cells)} grid cells')

    if not cells:
        print('No valid cells found, HDF5 file not created.')
        return 
    
    # Convert the list of cells to a single NumPy array
    stacked_array = np.stack(cells)  # Shape: (x, 1, 10, 10)

    # Create the HDF5 file
    base_name = os.path.splitext(os.path.basename(tiff_file))[0]
    output_file = os.path.join(output_directory, f'{base_name}.h5')

    with h5py.File(output_file, 'w') as h5file:
        # Write the stacked array to the HDF5 file
        h5file.create_dataset('cells', data=stacked_array, compression='gzip', compression_opts=9)
        print(f'dataset saved to {output_file}')

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
    print('finished')
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="input directory path")
    parser.add_argument('--output_dir', type=str, help="output directory path")
    parser.add_argument('--grid_size', type=int, help="snapshot size of square grid")

    args = parser.parse_args()
    
    process_directory(args.input_dir, args.grid_size, args.output_dir)


if __name__ == "__main__":
    main()