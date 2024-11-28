import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class PTHDataset(Dataset):
    def __init__(self, train_dir, target_dir):
        """
        Initialize dataset by organizing batches across folders.

        Args:
            root_dir (str): Root directory containing folders of data.
        """
        self.train_dir = train_dir
        self.target_dir = target_dir

        # Determine the number of batches (files) in each folder
        self.num_batches = len(sorted(os.listdir(self.train_dir))) # Assuming all folders have the same structure
        self.batch_files = [f"{i:07}.pth" for i in range(self.num_batches)]

    def __len__(self):
        """Returns the total number of batches across all folders."""
        return self.num_batches

    def __getitem__(self, idx):
        """
        Retrieves the batch across all folders at a given index.

        Args:
            idx (int): Index of the batch to load from each folder.

        Returns:
            torch.Tensor: Concatenated tensor across all folders.
        """

        train_file = os.path.join(self.train_dir, self.batch_files[idx])
        train = torch.load(train_file)
        target_file = os.path.join(self.target_dir, self.batch_files[idx])
        target = torch.load(target_file)

        # Concatenate along the second dimension (folder axis)
        return train, target

def get_dataloaders(directory, target_file, batch_size):
    # Create the dataset and dataloader
    dataset = PTHDataset(directory, target_file)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Usage example
if __name__ == "__main__":
    # Root directory containing the folders of batch data
    root_folder = "../data/final_train"
    target_folder = "../data/final_target"

    # Create DataLoader
    train_loader, _ = get_dataloaders(root_folder, target_folder, 1)

    # Iterate through the DataLoader
    for train, target in train_loader:
        print(f"Combined batch shape: {train.shape} {target.shape}")
        break
