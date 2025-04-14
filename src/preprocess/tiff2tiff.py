import os
import rasterio
import numpy as np
from rasterio.enums import Resampling

def mask_with_ground_truth(ground_truth_path, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Load the ground truth raster
    with rasterio.open(ground_truth_path) as gt_src:
        gt_data = gt_src.read(1)
        gt_profile = gt_src.profile
        nan_mask = np.isnan(gt_data)

    # Iterate through all raster files in input_folder
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue  # skip non-raster files

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile

            # Check dimensions match
            if data.shape != gt_data.shape:
                raise ValueError(f"Dimension mismatch: {filename} does not match ground truth.")

            # Apply NaN mask from ground truth
            masked_data = data.astype('float32')  # allow for NaNs
            masked_data[nan_mask] = np.nan

            # Update profile to support float32 and NaNs
            profile.update(dtype='float32', nodata=np.nan)

            # Write masked raster
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(masked_data, 1)

    print("All rasters masked and saved to:", output_folder)


# Example usage
ground_truth_path = '/media/data/ground_truth/ground_truth_norm.tif'
input_folder = '/media/data/iter3/raw_raster/mean_temp'
output_folder = '/media/data/iter3/proc_raster'

mask_with_ground_truth(ground_truth_path, input_folder, output_folder)
