import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MultiFileTensorDataset(Dataset):
    def __init__(self, x_dir, y_file_path):
        """
        Initialize dataset by loading multiple X tensor files and a single y tensor file
        
        Args:
            x_dir (str): Directory containing X tensor files
            y_file_path (str): Path to the y tensor file
        """
        self.y = torch.load(y_file_path)
        
        # Load and concatenate X tensor files
        x_files = sorted([os.path.join(x_dir, f) for f in os.listdir(x_dir) if f.endswith('.pth')])
        
        # Load and concatenate X tensors
        X_list = []
        for x_file in x_files:
            x_tensor = torch.load(x_file)
            X_list.append(x_tensor)
        
        self.X = torch.cat(X_list, dim=1)
        
        # Ensure correct tensor types
        self.X = self.X.float()
        self.y = self.y.float()
    
    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.y)
    
    def __getitem__(self, idx):
        """Retrieves a single sample and its corresponding label"""
        return self.X[idx], self.y[idx]

def create_dataloader(x_dir, y_file_path, batch_size=32, shuffle=True):
    """
    Create a DataLoader from multiple X tensor files and a single y tensor file
    
    Args:
        x_dir (str): Directory containing X tensor files
        y_file_path (str): Path to the y tensor file
        batch_size (int, optional): Number of samples per batch
        shuffle (bool, optional): Whether to shuffle the data
    
    Returns:
        torch.utils.data.DataLoader: DataLoader object
    """
    dataset = MultiFileTensorDataset(x_dir, y_file_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    return dataloader

def get_dataloaders(batch_size, directory, target_file):
    # Create the dataset and dataloader
    dataset = MultiFileTensorDataset(directory, target_file)

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
    
    # Create DataLoader
    train_loader = create_dataloader('../data/train_pth', '/home/thpc/workspace/pfas_map_ml/data/target_pth/HUC8_CA_PFAS_GTruth_Summa3.pth', batch_size=2)
    
    # Iterate through the DataLoader
    for batch_X, batch_y in train_loader:
        print(f"Batch features shape: {batch_X.shape}")
        print(f"Batch labels shape: {batch_y.shape}")
        break