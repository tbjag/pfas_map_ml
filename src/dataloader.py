# util bla blah blah
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class H5Dataset(Dataset):
    def __init__(self, directory, target_file):
        """
        Args:
            directory (str): Path to the directory containing HDF5 files.
            target_file (str): Path to the HDF5 file containing target data.
        """
        self.directory = directory
        self.h5_files = [
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')
        ]
        if not self.h5_files:
            raise ValueError(f"No HDF5 files found in directory: {directory}")
        self.num_files = len(self.h5_files)

        # Load target file lazily
        self.target_file = target_file
        with h5py.File(self.target_file, 'r') as h5:
            self.target_data = h5['cells'][:]  # Load the entire target data to memory
            if self.target_data.shape[1:] != (1, 10, 10):
                raise ValueError(f"Target data shape {self.target_data.shape[1:]} does not match (1, 10, 10)")

        # Validate that the number of samples matches across all files
        self.num_samples = self.target_data.shape[0]

    def __len__(self):
        """Returns the number of samples."""
        return self.num_samples

    def __getitem__(self, index):
        """
        Fetches the `index`th sample from all files and the corresponding target.
        """
        input_data = []
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as h5:
                # Read the `index`th sample from the dataset in each file
                data = h5['cells'][index]  # Shape: (1, 10, 10)
                input_data.append(data.squeeze())  # Remove the singleton dimension, shape: (10, 10)

        # Stack along a new dimension for NUM_H5_FILES
        input_data = torch.tensor(input_data, dtype=torch.float32)  # Shape: (NUM_H5_FILES, 10, 10)

        # Get the corresponding target sample
        target = torch.tensor(self.target_data[index], dtype=torch.float32)  # Shape: (1, 10, 10)

        return input_data, target


def get_dataloaders(batch_size, directory, target_file):
    # Create the dataset and dataloader
    dataset = H5Dataset(directory, target_file)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader