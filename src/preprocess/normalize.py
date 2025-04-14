import os
import rasterio
import numpy as np
from rasterio.plot import show
from rasterio.enums import Resampling

def normalize_raster(array):
    """Normalize a raster array to the range 0-1, ignoring NaN values."""
    mask = array == array  # Identify valid (non-NaN) values
    min_val = np.min(array[mask]) if np.any(mask) else 0
    max_val = np.max(array[mask]) if np.any(mask) else 1
    
    if max_val - min_val == 0:
        return np.zeros_like(array, dtype=np.float32)  # Avoid division by zero
    
    normalized_array = np.where(mask, (array - min_val) / (max_val - min_val), array)
    return normalized_array

def process_tiff_folder(input_folder, output_folder):
    """Process all TIFF files in a folder, normalizing them between 0 and 1 while preserving NaNs."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with rasterio.open(input_path) as src:
                profile = src.profile  # Preserve metadata
                data = src.read(1).astype(np.float32)  # Read first band
                
                # Preserve NaN values
                nan_mask = data == src.nodata
                normalized_data = normalize_raster(data)
                normalized_data[nan_mask] = src.nodata
                
                profile.update(dtype=rasterio.float32)  # Update data type
                
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(normalized_data, 1)
            
            print(f"Processed: {filename} -> {output_path}")


# Example usage
input_folder = "/media/data/ground_truth"
output_folder = "/media/data/ground_truth_norm"
process_tiff_folder(input_folder, output_folder)
