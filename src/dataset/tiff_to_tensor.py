import os
import argparse
import numpy as np
import rasterio
import torch
from typing import List, Tuple
from tqdm import tqdm

class TIFFProcessor:
    def __init__(self, output_directory: str, valid_locations_file: str = None, verbose: bool = False):
        self.output_directory = output_directory
        self.valid_locations_file = valid_locations_file
        self.verbose = verbose
        self.grid_size = 32  # Set grid size here
        os.makedirs(output_directory, exist_ok=True)

    def _load_valid_locations(self) -> List[Tuple[int, int, int, int]]:
        if not self.valid_locations_file or not os.path.exists(self.valid_locations_file):
            raise ValueError("Valid locations file not specified or does not exist.")
        
        valid_locations = []
        with open(self.valid_locations_file, 'r') as f:
            for line in f:
                start_row, start_col, end_row, end_col = map(int, line.strip().split(','))
                valid_locations.append((start_row, start_col, end_row, end_col))
        
        return valid_locations

    def _preprocess_cell(self, cell: np.ndarray) -> np.ndarray:
        processed_cell = cell.copy()
        
        band_data = processed_cell[0, :, :]
        
        # Check if the entire band is NaN
        if np.isnan(band_data).all():
            # If entire band is NaN, replace with zeros
            processed_cell[0, :, :] = np.zeros_like(band_data)
        else:
            # Calculate mean, ignoring NaN values
            band_mean = np.nanmean(band_data)
            
            # Replace NaN values with mean
            band_data[np.isnan(band_data)] = band_mean
            
            # Replace negative values with 0
            band_data[band_data < 0] = 0
            
            # Put the processed band back into the cell
            processed_cell[0, :, :] = band_data
        
        return processed_cell

    def process_tiff(self, tiff_file: str) -> int:
        cell_count = 0
        try:
            valid_locations = self._load_valid_locations()

            with rasterio.open(tiff_file) as src:
                tiff_data = src.read()
                with tqdm(total=len(valid_locations), 
                          desc=f"Processing {os.path.basename(tiff_file)}", 
                          disable=not self.verbose,
                          unit='cell') as pbar:
                    
                    for start_row, start_col, end_row, end_col in valid_locations:
                        cell = tiff_data[:, start_row:end_row, start_col:end_col]
                        processed_cell = self._preprocess_cell(cell)
                        
                        output_path = os.path.join(self.output_directory, f'{cell_count:07}.pth')
                        cell_tensor = torch.tensor(processed_cell, dtype=torch.float32)
                        
                        if os.path.exists(output_path):
                            existing_tensor = torch.load(output_path)
                            updated_tensor = torch.cat((existing_tensor, cell_tensor), dim=0)
                            torch.save(updated_tensor, output_path)
                        else:
                            torch.save(cell_tensor, output_path)
                        
                        cell_count += 1
                        
                        pbar.update(1)

        except Exception as e:
            print(f"Failed to process {tiff_file}: {e}")
        
        return cell_count

    def process_directory(self, input_directory: str) -> List[str]:
        tiff_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(input_directory)
            for file in files if file.endswith('.tiff')
        ]

        total_cells = 0
        processed_files = []

        with tqdm(total=len(tiff_files), 
                  desc="Processing TIFF Files", 
                  disable=not self.verbose,
                  unit='file') as pbar:
            for tiff_file in tiff_files:
                cells_in_file = self.process_tiff(tiff_file)
                
                if cells_in_file > 0:
                    processed_files.append(tiff_file)
                    total_cells += cells_in_file
                
                pbar.update(1)

        if self.verbose:
            print(f"\nProcessing complete:")
            print(f"Total files processed: {len(processed_files)}")
            print(f"Total cells extracted: {total_cells}")
        
        return processed_files

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process TIFF files into grid-based tensors using valid cell locations.')
    
    parser.add_argument('-i', '--input-dir', 
                        type=str, 
                        help='Input directory containing TIFF files')
    
    parser.add_argument('-o', '--output-dir', 
                        type=str, 
                        help='Output directory for processed tensor files')
    
    parser.add_argument('-l', '--locations-file', 
                        type=str, 
                        required=True,
                        help='Path to file containing valid cell locations')
    
    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Enable verbose output and progress bars')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    processor = TIFFProcessor(
        output_directory=args.output_dir, 
        valid_locations_file=args.locations_file, 
        verbose=args.verbose
    )
    processor.process_directory(args.input_dir)

if __name__ == "__main__":
    main()