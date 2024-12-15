## Commands to run 

Overall we take various geographic data and process them into pytorch tensors. We want process them as rasters. So we convert csvs to rasters and then turn them into pth files. We also resize and crop existing rasters to fit the correct dimensions.


The final data is stored as a multi-dimensional matrix as follows: NUMBER_DIMENSIONS X GRID_SIZE X GRID_SIZE. The grid size the size of the snapshot of the raster and usually the pixel size.
The number of dimensions is the number of raster files processed. 

Some notes: There are issues with precision and cropping with numpy and rasterio. To have consistent grids, we create a list of all valid grids in the ground truth data and then use that list to process all other files. 

Command to create valid locations: `python create_valid_locations.py -r /media/data/ground_truth/ground_truth.tiff -o /media/data/ground_truth/ground_truth_valid_locations.txt -v`

### Tiff to Tiff

Taking a raster and overlaying our shapefile to crop to the PFAS output

`python tiff_to_tiff.py --input_dir /media/data/raw/tiffs/ --shapefile /media/data/const_shapes/border_outline.shp --output_dir /media/data/iter2/processed_tiffs/`


### CSV to Tiff

- ensure that each csv is sorted into categorical or discrete categories
- for categorical: set the target value in the gdf to 1
- for discrete: remove setting the target value, ensure your target value is saved as the column name in the csv TODO: can parameterize

TODO add command here

### Processed Tiff to pytorch tensor

If running for the first time, set variable first to true. Otherwise set to false if tensors already exist.

` python tiff_to_tensor.py -i /media/data/iter2/test_tiffs/ -o /media/data/TEST/pth -l /media/data/ground_truth/ground_truth_valid_locations.txt -v`

grid size as a default is set to 32x32.