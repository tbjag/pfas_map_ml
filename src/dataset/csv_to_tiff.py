import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin

import argparse
import os
from tqdm import tqdm

from constants import COORD_SYSTEM, BUFFER_DIV, NEGATIVE_CONST

class Config:
    def __init__(self, input_dir, shp_filepath, output_dir, buffer_size, is_categorical) -> None:
        self.input_dir = input_dir
        self.shp = get_shp(shp_filepath)
        self.output_dir = output_dir
        self.buffer_size = buffer_size / BUFFER_DIV
        self.is_categorical = is_categorical

def queue_files(dir_path):
    file_info  = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]  # Get filename without extension
                file_info.append((full_path, file_name))
    return file_info

def check_paths(dir_path, shp_path, output_dir):
    check = True

    if not os.path.isdir(dir_path):
        check = False
        print(f'csv folder {dir_path} does not exist')

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

def set_geometry(csv_filepath, config: Config):
    df = pd.read_csv(csv_filepath)
    # TODO should always contain lon, lat as data
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:32633")
    gdf['geometry'] = gdf['geometry'].buffer(config.buffer_size)  # Buffer the points 

    # TODO everything in this script contains target rn, uncomment if target already exists
    gdf['target'] = 1

    return gdf

def transfrom_to_raster(gdf, config, filename):
    # Calculate raster dimensions based on shapefile bounds
    xmin, ymin, xmax, ymax = config.shp.total_bounds
    pixel_size = 0.001  # Adjust as needed
    width = int((xmax - xmin) / pixel_size)
    height = int((ymax - ymin) / pixel_size)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    # Create a mask for California shape
    california_raster_mask = geometry_mask(
        config.shp['geometry'], 
        transform=transform, 
        invert=True,  # Invert to get True inside the shape
        out_shape=(height, width)
    )

    shapes_with_target = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['target'])]
    buffered_raster = rasterize(
        shapes_with_target, 
        out_shape=(height, width), 
        transform=transform, 
        fill=0,  # Default fill value
        dtype=rasterio.float32
    )

    final_raster = np.full((height, width), NEGATIVE_CONST, dtype=rasterio.float32)
    final_raster[california_raster_mask] = 0
    
    # Overlay buffered points on top of California shape
    # This ensures buffered points are visible only inside California
    mask_and_buffer = (california_raster_mask & (buffered_raster > 0))
    final_raster[mask_and_buffer] = buffered_raster[mask_and_buffer]

    new_raster_path = os.path.join(config.output_dir, filename + '.tiff')

    with rasterio.open(
        new_raster_path, 
        'w',
        driver='GTiff',
        height=final_raster.shape[0],
        width=final_raster.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4269",
        transform=transform,
        nodata=NEGATIVE_CONST
    ) as dst:
        dst.write(final_raster, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help="input directory path", required=True)
    parser.add_argument('-s', '--shp_filepath', type=str, help="shapefile path", required=True)
    parser.add_argument('-o', '--output_dir', type=str, help="output directory path", required=True)
    parser.add_argument('-b', '--buffer_size', type=int, help="buffer size radius in km", required=True)

    args = parser.parse_args()

    check_paths(args.input_dir, args.shp_filepath, args.output_dir)
    config = Config(args.input_dir, args.shp_filepath, args.output_dir, args.buffer_size, args.is_categorical)
    
    filepaths = queue_files(config.input_dir)

    for filepath, filename in tqdm(filepaths, desc=f"processing {len(filepaths)} files"):
        gdf = set_geometry(filepath, config)
        transfrom_to_raster(gdf, config, filename)

    print('finished')

if __name__ == "__main__":
    main()