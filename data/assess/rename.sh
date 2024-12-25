#!/bin/bash

# Directory containing the CSV files
directory="./"

# Loop over all CSV files ending in ...0.csv in the directory
for file in "$directory"TCPR123_year*; do
    # Check if the file exists
    if [[ -f "$file" ]]; then
        # Create a temporary file for the updated CSV
        temp_file=$(mktemp)

        # Replace the column names using sed and save to the temporary file
        sed '1s/gm_longitude/lon/; 1s/gm_latitude/lat/; 1s/gm_result/target/' "$file" > "$temp_file"

        # Overwrite the original file with the modified content
        mv "$temp_file" "$file"

        echo "Updated columns in $file"
    else
        echo "No matching files found."
    fi
done