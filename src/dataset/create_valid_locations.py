import os
import argparse
import numpy as np
import rasterio

def validate_raster(raster_path: str, grid_size: int, output_path: str = None, verbose: bool = False):
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        output_path = f"{base_name}_valid_cells.txt"

    valid_locations = []

    try:
        # Open the raster file
        with rasterio.open(raster_path) as src:
            # Read the raster data
            tiff_data = src.read()
            _, original_height, original_width = tiff_data.shape

            # Track total and valid cell counts
            total_cells = 0
            valid_cell_count = 0

            # Iterate through the raster in grid_size steps
            for i in range(0, original_height, grid_size):
                for j in range(0, original_width, grid_size):
                    # Extract cell
                    cell = tiff_data[:, i:i+grid_size, j:j+grid_size]
                    
                    # Increment total cells
                    total_cells += 1

                    # Check if cell is valid
                    if (cell.shape == (1, grid_size, grid_size) and 
                        not np.any(np.isnan(cell)) and 
                        not np.any(cell < 0)):
                        
                        # Add to valid locations
                        valid_locations.append((i, j, i+grid_size, j+grid_size))
                        valid_cell_count += 1

            # Write valid locations to file
            with open(output_path, 'w') as f:
                for loc in valid_locations:
                    f.write(f"{loc[0]},{loc[1]},{loc[2]},{loc[3]}\n")

            # Verbose output
            if verbose:
                print(f"Raster: {raster_path}")
                print(f"Grid Size: {grid_size}x{grid_size}")
                print(f"Total Cells: {total_cells}")
                print(f"Valid Cells: {valid_cell_count}")
                print(f"Valid Locations saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {raster_path}: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Validate cells in a raster file and generate valid cell locations.')
    
    parser.add_argument('-r', '--raster', 
                        type=str, 
                        required=True,
                        help='Path to the input raster file')
    
    parser.add_argument('-g', '--grid-size', 
                        type=int, 
                        default=32, 
                        help='Size of grid cells to validate (default: 32)')
    
    parser.add_argument('-o', '--output', 
                        type=str, 
                        help='Path to save valid cell locations (optional)')
    
    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    validate_raster(
        raster_path=args.raster, 
        grid_size=args.grid_size,
        output_path=args.output,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()