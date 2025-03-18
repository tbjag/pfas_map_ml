import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
import numpy as np

def resize_rasters(input_dir, output_dir, target_resolution):
    """
    Resizes all raster files in the input directory to the specified resolution.
    
    Args:
        input_dir (str): Directory containing input raster files.
        output_dir (str): Directory to save resized rasters.
        target_resolution (tuple): (target_width, target_height) in pixels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with rasterio.open(input_path) as src:
                transform, width, height = calculate_default_transform(
                        src.crs, src.crs, target_resolution[0], target_resolution[1], *src.bounds
                    )
                kwargs = src.meta.copy()
                kwargs.update({
                    "height": height,
                    "width": width,
                    "transform": transform
                })
                
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        resampled_data = src.read(i, out_shape=(height, width), resampling=Resampling.bilinear)
                        dst.write(resampled_data, i)
            
            print(f"Resized and saved: {output_path}")

# Example usage
input_directory = "/media/data/raw/nico_tiffs"
output_directory = "/media/data/raw/nico_resize"
target_size = (9469, 10215)  # Target resolution in 9469 10215
resize_rasters(input_directory, output_directory, target_size)
