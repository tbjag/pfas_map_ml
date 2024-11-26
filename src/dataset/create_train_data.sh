#!/bin/bash

# Set the variables
INPUT_DIR_CSV="../../data/og_csv"
SHP_FILEPATH="../../HUC8_CA_PFAS_GTruth.shp"
OUTPUT_DIR_TIFF="../../data/tiff_temp"
BUFFER_SIZE=10

INPUT_DIR_BIG_TIFF="../../data/tiff_temp"
OUTPUT_DIR_H5="../../data/train"
GRID_SIZE=10

# Run the Python scripts with the specified variables
python csv_to_tiff.py \
  --input_dir="$INPUT_DIR_CSV" \
  --shp_filepath="$SHP_FILEPATH" \
  --output_dir="$OUTPUT_DIR_TIFF" \
  --buffer_size="$BUFFER_SIZE"

python tiff_to_h5.py \
  --input_dir="$INPUT_DIR_BIG_TIFF" \
  --output_dir="$OUTPUT_DIR_H5" \
  --grid_size="$GRID_SIZE"
