import os
import rasterio
import numpy as np
from rasterio.enums import Resampling

# Set this to your input and output folder
input_folder = '/media/data/iter3/raw_raster/rivers'
output_folder = '/media/data/iter3/raw_raster/rivers_nan'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with rasterio.open(input_path) as src:
            # Read with mask to catch nodata as masked values
            data = src.read(1, masked=True)

            # Convert masked values to NaN
            data_with_nan = data.filled(np.nan).astype(np.float32)

            # Update metadata to reflect float32 and nodata as nan
            profile = src.profile
            profile.update(dtype='float32', nodata=np.nan)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data_with_nan, 1)

        print(f"Converted and saved: {output_path}")
