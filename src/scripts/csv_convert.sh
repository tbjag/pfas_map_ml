#!/bin/bash

# CSV to TIFF conversion
INPUT_DIR_CSV="/media/data/raw/csv_categorical" # CHANGE 
SHP_FILEPATH="/media/data/const_shapes/border_outline.shp"
OUTPUT_DIR_TIFF="/media/data/iter3/processed_csv_cat" # CHANGE
BUFFER_SIZE=10

OUTPUT_DIR="media/data/iter2/train"

cd ../dataset

python csv_to_tiff.py \
  --input_dir="$INPUT_DIR_CSV" \
  --shp_filepath="$SHP_FILEPATH" \
  --output_dir="$OUTPUT_DIR_TIFF" \
  --buffer_size="$BUFFER_SIZE"

# python tiff_to_tensor.py \
#   --input_dir="$OUTPUT_DIR_TIFF" \
#   --output_dir="$OUTPUT_DIR"
