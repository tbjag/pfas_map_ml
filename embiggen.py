import os
import rasterio
from rasterio.enums import Resampling

#Create Stacked Tensors for each file in selected directory
def process_directory(directory, output_directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                tiff_file = os.path.join(root, file)
                try:
                    embiggen(tiff_file, output_directory)
                except Exception as e:
                    print(f"Failed to process {tiff_file}: {e}")
    
    print('done')

def embiggen(filepath, outputdir):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_file = os.path.join(outputdir, f'{base_name}.tiff')

    # Open the source TIFF file
    with rasterio.open(filepath) as src:
        # Calculate new shape for doubling
        new_width = src.width * 10
        new_height = src.height * 10

        # Calculate new transform (to adjust the resolution)
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),  # scaling in the X direction
            (src.height / new_height)  # scaling in the Y direction
        )
        
        # Read and resample the data to the new shape
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear  # you can use other resampling methods if needed
        )
        
        # Update metadata
        profile = src.profile
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'driver': 'GTiff'
        })

        # Write the output to a new TIFF file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data)
    
    print(f'finished {filepath}')




def main():
    input_dir = 'og2'
    output_dir = 'large_tiff_data3'
    
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()