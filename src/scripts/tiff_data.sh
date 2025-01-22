#!/bin/bash

# TIFF to TIFF conversion
INPUT_DIR_TIFF="/media/data/raw/" # TODO 
SHP_FILEPATH="/media/data/const_shapes/border_outline.shp"
OUTPUT_DIR_TIFF="/media/data/iter2/processed_tiff" 

OUTPUT_DIR="media/data/iter2/train"

python tiff_to_tiff.py \
  --input_dir="$INPUT_DIR_CSV" \
  --shp_filepath="$SHP_FILEPATH" \
  --output_dir="$OUTPUT_DIR_TIFF"

python tiff_to_tensor.py \
  --input_dir="$OUTPUT_DIR_TIFF" \
  --output_dir="$OUTPUT_DIR" 