## Commands to run 

### Tiff to Tiff

Taking a raster and overlaying our shapefile to crop to the PFAS output


`python tiff_to_tiff.py --input_dir /media/data/raw/tiffs/ --shapefile /media/data/const_shapes/border_outline.shp --output_dir /media/data/iter2/processed_tiffs/`


### CSV to Tiff

- ensure that each csv is sorted into categorical or discrete categories
- for categorical: set the target value in the gdf to 1
- for discrete: remove setting the target value, ensure your target value is saved as the column name in the csv

### Processed Tiff to pytorch tensor

If running for the first time, set variable first to true. Otherwise set to false if tensors already exist.

`python train_to_data.py --input_dir /media/data/iter2/processed_tiffs/ --output_dir /media/data/iter2/train`

grid size as a default is set to 32x32.