# Set the variables
INPUT_DIR="/media/data/iter3/proc_raster/csvs"
OUTPUT_DIR="/media/data/iter3/train/csvs"
LOCATIONS="/media/data/ground_truth/grid32.txt"
GRID_SIZE=32

cd ../dataset

# Run the Python scripts with the specified variables
python tiff_to_tensor.py \
  -i="$INPUT_DIR" \
  -o="$OUTPUT_DIR" \
  -l="$LOCATIONS" \
  -v

