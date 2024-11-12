import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from rasterio.mask import mask

import argparse
import yaml
import os

from constants import COORD_SYSTEM, BUFFER_DIV, NEGATIVE_CONST

class Config:
    def __init__(self, input_dir, shp_filepath, output_dir, buffer_size) -> None:
        self.input_dir = input_dir
        self.shp = get_shp(shp_filepath)
        self.output_dir = output_dir
        self.buffer_size = buffer_size / BUFFER_DIV

def load_config(yaml_filepath):
    with open(yaml_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def queue_files(dir_path):
    file_info  = []
    for root, _, files in os.walk(dir_path):
        for file in files:
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

def set_geometry(csv_filepath, filename, config: Config):
    df = pd.read_csv(csv_filepath)
    # TODO should always contain lon, lat as data
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:32633")
    gdf['geometry'] = gdf['geometry'].buffer(config.buffer_size)  # Buffer the points 5km

    # TODO everything in this script contains target rn
    gdf['target'] = 2

    return gdf

def transfrom_to_raster(gdf, config, filename):
    xmin, ymin, xmax, ymax = config.shp.total_bounds
    pixel_size = 0.01  # Adjust as needed
    width = int((xmax - xmin) / pixel_size)
    height = int((ymax - ymin) / pixel_size)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    shapes_with_target = [(geom, value) for geom, value in zip(gdf['geometry'], gdf['target'])] # TODO assumes target col name is target
    
    buffered_raster = rasterize(shapes_with_target, out_shape=(height, width), transform=transform, fill=0, dtype=rasterio.float32)

    california_raster = np.zeros((height, width), dtype=np.uint8)
    california_raster_mask = geometry_mask(config.shp['geometry'], transform=transform, invert=True, out_shape=(height, width))
    california_raster[california_raster_mask] = 1  # Set pixels inside California border to 0

    combined_raster = np.maximum(buffered_raster, california_raster)

    new_raster_path = os.path.join(config.output_dir, filename)
    new_raster_path += '.tiff'

    with rasterio.open(
        new_raster_path, 'w',
        driver='GTiff',
        height=combined_raster.shape[0],
        width=combined_raster.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4269",
        transform=transform,
    ) as dst:
        dst.write(combined_raster, 1)

    return new_raster_path

def add_null_val(raster_path, config):
    # Mask out values outside California
    ca_shapes = [feature["geometry"] for feature in config.shp.__geo_interface__["features"]]
    with rasterio.open(raster_path, 'r+') as src:
        # Read the entire raster
        data = src.read(1)

        # Generate mask
        mask_array, _ = mask(src, ca_shapes, crop=False)

        # Set values outside the mask to NoData
        nodata_value = src.nodata if src.nodata is not None else NEGATIVE_CONST
        data[mask_array[0] == False] = nodata_value

        # Update metadata to include NoData value
        out_meta = src.meta.copy()
        out_meta.update({"nodata": nodata_value})

        src.write(data, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the YAML configuration file")
    args = parser.parse_args()
    config_yaml = load_config(args.config)
    check_paths(config_yaml['input_dir'], config_yaml['shp_filepath'], config_yaml['output_dir'])
    # Access the config values
    config = Config(config_yaml['input_dir'], config_yaml['shp_filepath'], config_yaml['output_dir'], config_yaml['buffer_size'])
    
    filepaths = queue_files(config.input_dir)

    for filepath, filename in filepaths:
        print(f'working on {filename}.csv')
        gdf = set_geometry(filepath, filename, config)
        raster_path = transfrom_to_raster(gdf, config, filename)
        add_null_val(raster_path, config)
        print(f'saved {filename}.tiff')

    print('finished')

if __name__ == "__main__":
    main()