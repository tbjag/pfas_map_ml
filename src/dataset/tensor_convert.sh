# Set the variables
INPUT_DIR="/media/data/iter2/processed_tiffs"
OUTPUT_DIR="/media/data/iter2/train"
LOCATIONS="/media/data/ground_truth/ground_truth_valid_locations.txt"
GRID_SIZE=32

# Run the Python scripts with the specified variables
python tiff_to_tensor.py \
  -i="$INPUT_DIR" \
  -o="$OUTPUT_DIR" \
  -l="$LOCATIONS" \
  -v

