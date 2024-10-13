import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import features
from rasterio.plot import show
from rasterio.mask import mask
import argparse
import warnings
from rasterio.warp import reproject, Resampling
import tempfile
import shutil
import yaml

from constants import COORD_SYSTEM, TARGET_WIDTH, TARGET_HEIGHT, TARGET_RESOLUTION, MIN_LON, MAX_LAT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the YAML configuration file")

    args = parser.parse_args()

    # Load the configuration from the YAML file
    config = load_config(args.config)

    # Access the config values
    args.csv_filepath = config['csv_filepath']
    args.shp_filepath = config['shp_filepath']
    args.buffer_size = config['buffer_size']
    args.output_filename = config['output_filename']
    args.output_folder = config['output_folder']
    args.is_folder = config['input_is_folder']

    check_paths(args)

    if args.is_folder:
        filepaths = queue_files(args.csv_filepath)
    else:
        filepaths = [args.csv_filepath]
        
 
    for filepath in filepaths:
        args.csv_filepath = filepath
        buffer_geometry = get_geometry(args)
        cal_shape = get_shp(args)

        combined_raster, transform = transfrom_to_raster(buffer_geometry, cal_shape)
        raster_path = save_raster_combine(args, combined_raster, transform)
        add_null_val(raster_path, cal_shape)
        transform_tif(raster_path)

    print('finished')

def load_config(yaml_filepath):
    with open(yaml_filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def queue_files(dir_path):
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def check_paths(args):
    csv_path, shp_path, output_path, is_folder= args.csv_filepath, args.shp_filepath, args.output_folder, args.is_folder
    check = True

    if is_folder:
        if not os.path.isdir(csv_path):
            check = False
            print('csv folder does not exist')
    else:
        if not os.path.isfile(csv_path):
            check = False
            print('csv_filepath does not exist')
    if not os.path.isfile(shp_path):
        check = False
        print('shp_filepath does not exist')
    if not os.path.isdir(output_path):
        check = False
        print('combined_folder does not exist')

    if not check:
        exit()

def get_geometry(args):
    df = pd.read_csv(args.csv_filepath)

    check_lat, check_lon = True, True
    for col in df.columns:
        if col in ['long', 'lon', 'LON', 'LONG', 'longitude', 'LONGITUDE']:
            check_lon = False
            df.rename(columns={col: 'longitude'}, inplace=True)

        if col in ['lat', 'LAT', 'latitude', 'LATITUDE']:
            check_lat = False
            df.rename(columns={col: 'latitude'}, inplace=True)

    if check_lon:
        print('could not find longitude column')
        exit()

    if check_lat:
        print('could not find latitude column')
        exit()

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=COORD_SYSTEM) # leave acomment about crs

    coord_transform = args.buffer_size / 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        geo_buf = gdf['geometry'].buffer(coord_transform)

    return geo_buf

def get_shp(args):
    california_border = gpd.read_file(args.shp_filepath)
    california_border.crs = COORD_SYSTEM
    return california_border

def transfrom_to_raster(buffer_geometry, cal_border):
    # Determine raster extent and pixel size
    xmin, ymin, xmax, ymax = cal_border.total_bounds
    pixel_size = 0.01  # Adjust as needed
    width = int((xmax - xmin) / pixel_size)
    height = int((ymax - ymin) / pixel_size)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    # Create raster for buffer zones
    buffered_raster = np.zeros((height, width), dtype=np.uint8)
    buffered_raster_mask = features.geometry_mask(buffer_geometry, transform=transform, invert=True, out_shape=(height, width))
    buffered_raster[buffered_raster_mask] = 2  # Set pixels inside buffered geometries to 2

    # Create raster for California border
    california_raster = np.zeros((height, width), dtype=np.uint8)
    california_raster_mask = features.geometry_mask(cal_border['geometry'], transform=transform, invert=True, out_shape=(height, width))
    california_raster[california_raster_mask] = 1  # Set pixels inside California border to 1

    return np.maximum(buffered_raster, california_raster), transform

def save_raster_combine(args, combined_raster, transform):
    combined_raster_path = os.path.join(args.output_folder, f'{int(args.buffer_size)}km_' + args.output_filename)

    with rasterio.open(
        combined_raster_path, 'w',
        driver='GTiff',
        height=combined_raster.shape[0],
        width=combined_raster.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=COORD_SYSTEM,
        transform=transform,
    ) as dst:
        dst.write(combined_raster, 1)

    return combined_raster_path

def add_null_val(raster_path, ca_shape):
    ca_shapes = [feature["geometry"] for feature in ca_shape.__geo_interface__["features"]]
    with rasterio.open(raster_path, 'r+') as src:
        # Read the entire raster
        data = src.read(1)

        # Generate mask
        mask_array, _ = mask(src, ca_shapes, crop=False)

        # Check if NoData value exists, if not, set a default
        nodata_value = src.nodata if src.nodata is not None else 0  # Or any suitable default

        # Set values outside the mask to NoData
        data[mask_array[0] == False] = nodata_value

        # Update metadata to include NoData value
        out_meta = src.meta.copy()
        out_meta.update({"nodata": nodata_value})

        src.write(data, 1)

def transform_tif(raster_path):
    with rasterio.open(raster_path) as src:
        # Create a temporary file to hold the transformed raster
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
        
        with rasterio.open(raster_path) as src:
            # Calculate the target transform
            transform = from_origin(MIN_LON, MAX_LAT, TARGET_RESOLUTION[0], TARGET_RESOLUTION[1])

            # Define the metadata for the output file
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': COORD_SYSTEM,
                'transform': transform,
                'width': TARGET_WIDTH,
                'height': TARGET_HEIGHT,
                'driver': 'GTiff'
            })

            # Create the temporary output file with the updated metadata
            with rasterio.open(temp_output.name, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=COORD_SYSTEM,
                        resampling=Resampling.nearest
                    )
        
        # Replace the original file with the transformed file
        shutil.move(temp_output.name, raster_path)

if __name__ == "__main__":
    main()