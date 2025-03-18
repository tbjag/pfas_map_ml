import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
import argparse
import os

from constants import COORD_SYSTEM

def queue_files(dir_path):
    file_info  = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.tif'):
                full_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]  # Get filename without extension
                file_info.append((full_path, file_name))
    return file_info

def check_paths(input_path, shp_path, output_dir):
    check = True

    if not os.path.isdir(input_path):
        check = False
        print(f'tiff folder {input_path} does not exist')

    if not os.path.isfile(shp_path):
        check = False
        print(f'shp_filepath {shp_path} does not exist')

    if not os.path.isdir(output_dir):
        check = False
        print(f'output folder {output_dir} does not exist')

    if not check:
        exit()

def get_shp(shp_filepath):
    california_border = gpd.read_file(shp_filepath)
    california_border.crs = COORD_SYSTEM
    return california_border

def mask_tiff_with_shapefile(input_tiff, output_tiff, shapefile):

    with rasterio.open(input_tiff, 'r') as src:
        out_meta = src.meta.copy()

        # Perform masking (do NOT pass np.nan in mask function)
        out_image, out_transform = mask(src, shapefile.geometry, crop=False, nodata=None)

        # Convert masked raster to float and set NaN where original NoData was present
        out_image = out_image.astype(np.float32)
        if src.nodata is not None:
            out_image[out_image == src.nodata] = np.nan  # Replace original nodata with NaN

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": np.nan  # Set metadata to indicate NaN nodata
        })

        # Write masked raster
        with rasterio.open(output_tiff, 'w', **out_meta) as dest:
            dest.write(out_image)

    print(f"Masked TIFF saved to {output_tiff}")

def main():
    parser = argparse.ArgumentParser(description="Mask a TIFF file using a shapefile and handle NaN values")
    parser.add_argument('--input_dir', type=str, required=True, help="Input TIFF file path")
    parser.add_argument('--shapefile', type=str, required=True, help="Shapefile path")
    parser.add_argument('--output_dir', type=str, required=True, help="Output masked TIFF file path")
    
    args = parser.parse_args()
    check_paths(args.input_dir, args.shapefile, args.output_dir)
    print(args.input_dir)
    shape = get_shp(args.shapefile)
    
    for input_path, file_name in queue_files(args.input_dir):
        output_path = os.path.join(args.output_dir, file_name + '.tif')
        mask_tiff_with_shapefile(input_path, output_path, shape)


if __name__ == "__main__":
    main()