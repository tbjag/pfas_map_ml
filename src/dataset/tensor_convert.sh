# Set the variables
INPUT_DIR_CSV="../../data/tiff"
OUTPUT_DIR_TIFF="../../data/train"
GRID_SIZE=32

# Run the Python scripts with the specified variables
python train_to_data.py \
  --input_dir="$INPUT_DIR_CSV" \
  --output_dir="$OUTPUT_DIR_TIFF" \
  --grid_size=$GRID_SIZE

# for single PFAS files
# python target_to_data.py \
#   --input_dir="$INPUT_DIR_CSV" \
#   --output_dir="$OUTPUT_DIR_TIFF" \
#   --grid_size=$GRID_SIZE
