import rasterio
import h5py
import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import random

## take in h5 file directory, length of data

## create permutations

## access h5 files

## create batches which include all files

## create same batch for target file

def get_shuffled_indexes(data_length, batch_size):
    """
    Generates shuffled indexes in batches dynamically to save memory.
    Args:
        data_length (int): Total length of the data.
        batch_size (int): Number of samples per batch.
    Yields:
        list: Shuffled indexes for a batch.
    """
    all_indexes = list(range(data_length))
    random.shuffle(all_indexes)  # Shuffle the full range in place
    for i in range(0, data_length, batch_size):
        yield all_indexes[i : i + batch_size]


def process_train_data(files, output_dir, data_length, batch_size=64):
    """
    Processes training data by accessing H5 files partially, creating batches, and saving them.
    Args:
        files (list): List of file paths to H5 files.
        output_dir (str): Directory to save the batches.
        data_length (int): Total length of data for indexing.
        batch_size (int): Number of samples per batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_files = len(files)

    # Calculate total data length from the first file
    with h5py.File(files[0], "r") as f:
        assert data_length == f["cells"].shape[0], "Data length mismatch."

    # Process batches dynamically
    batch_idx = 0
    for shuffled_indexes in get_shuffled_indexes(data_length, batch_size):
        # Initialize batch storage
        batch_data = np.zeros((len(shuffled_indexes), num_files, 10, 10), dtype=np.float32)

        for file_idx, file in enumerate(files):
            with h5py.File(file, "r") as f:
                # Read only the required indexes from the current file
                partial_data = f["cells"][shuffled_indexes, :, :, :]
                batch_data[:, file_idx, :, :] = partial_data[:, 0, :, :]  # Remove singleton dimension

        # Convert to tensor and save
        batch_tensor = torch.tensor(batch_data)
        torch.save(batch_tensor, os.path.join(output_dir, f"batch_{batch_idx + 1}.pt"))
        batch_idx += 1

    print(f"Saved {batch_idx} training batches to {output_dir}")

def process_target_data(input_file, output_file, indexes):
    # 
    pass



def get_filepaths(directory):
    files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                files.append(os.path.join(root, file))
                
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_train', type=str, help="input directory path for training data") # TODO fix descriptions
    parser.add_argument('--output_dir_train', type=str, help="output directory path")
    # parser.add_argument('--input_file_target', type=str, help="input directory path for target")
    # parser.add_argument('--output_file_target', type=str, help="output directory path")

    DATA_LENGTH = 965866 # TODO change this
    # shuffled_indexes = np.random.permutation(DATA_LENGTH)

    args = parser.parse_args()
    files = get_filepaths(args.input_dir_train)
    print('MYAH')
    process_train_data(files, args.output_dir_train, DATA_LENGTH)


if __name__ == "__main__":
    main()