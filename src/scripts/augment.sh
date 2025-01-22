#!/bin/bash

# CSV to TIFF conversion
TRAIN_DIR="/media/data/iter3/train" # CHANGE 
TARGET_DIR="/media/data/iter3/target" # CHANGE 

cd ../dataset

python augment.py \
  --train_dir="$TRAIN_DIR" \
  --target_dir="$TARGET_DIR"