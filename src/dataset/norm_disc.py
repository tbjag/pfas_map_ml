import pandas as pd
import numpy as np
import os
import argparse

def queue_files(dir_path):
    file_info  = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_info.append((full_path, file))
    return file_info

def check_paths(input_dir, output_dir):
    check = True

    if not os.path.isdir(input_dir):
        check = False
        print(f'csv folder {input_dir} does not exist')

    if not os.path.isdir(output_dir):
        check = False
        print(f'output folder {output_dir} does not exist')

    if not check:
        exit()

def normalize_save(paths, output_dir):
    for path, filename in paths:
        df = pd.read_csv(path)
        # assumes we have a column named 'target'
        col = df['target']
        norm_col = (col - col.min()) / (col.max() - col.min())

        # Create a new DataFrame with normalized target
        new_df = df[['lon', 'lat']].copy()  # Use .copy() to ensure a new DataFrame is created
        new_df['target'] = norm_col  # No SettingWithCopyWarning here

        # Save the new DataFrame to the output directory
        new_df.to_csv(os.path.join(output_dir, filename), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="input directory path")
    parser.add_argument('--output_dir', type=str, help="output directory path")

    args = parser.parse_args()
    check_paths(args.input_dir, args.output_dir)

    paths = queue_files(args.input_dir)

    normalize_save(paths, args.output_dir)

    print('finished')

if __name__ == "__main__":
    main()
