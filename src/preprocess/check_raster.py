import os
import pandas as pd
import numpy as np
import rasterio

def process_raster(raster_path):
    try:
        with rasterio.open(raster_path) as src:
            resolution = src.res
            height = src.height
            width = src.width
            raster_data = src.read(1)
            raster_data = np.ma.masked_equal(raster_data, src.nodata)

            unique_values = np.unique(raster_data.compressed())
            min_value = np.nanmin(unique_values)
            max_value = np.nanmax(unique_values)
            unique_count = len(unique_values)

            return {
                "file_name": os.path.basename(raster_path),
                "resolution_x": resolution[0],
                "resolution_y": resolution[1],
                "height": height,
                "width": width,
                "min_value": min_value,
                "max_value": max_value,
                "unique_count": unique_count
            }
    except Exception as e:
        print(f"Error processing {raster_path}: {e}")
        return None

def process_folder(folder_path, output_csv):
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".tiff"):
            raster_path = os.path.join(folder_path, file_name)
            print(f"Processing: {raster_path}")
            raster_info = process_raster(raster_path)
            if raster_info:
                results.append(raster_info)

    if results:
        df = pd.DataFrame(results)
        file_exists = os.path.isfile(output_csv)
        df.to_csv(output_csv, mode='a', index=False, header=not file_exists)
        print(f"\nResults {'appended to' if file_exists else 'saved to'}: {output_csv}")

if __name__ == "__main__":
    folder_path = '/media/data/iter3_temp/processed_csv_cat'
    output_csv = 'results_csvs.csv'
    process_folder(folder_path, output_csv)
