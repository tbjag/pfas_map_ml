import os
import torch

def check_pth_file_sizes(folder_path):
    """
    Checks if all .pth files in a folder have the same size.

    Args:
        folder_path (str): Path to the folder containing .pth files.

    Returns:
        None
    """
    file_sizes = []
    pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]

    if not pth_files:
        print("No .pth files found in the folder.")
        return

    print(f"Found {len(pth_files)} .pth files. Checking sizes...")

    for file_name in pth_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Load the .pth file as a tensor dictionary
            data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
            
            file_sizes.append(data.shape[0])
        
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return

    # Check if all sizes are equal
    unique_sizes = set(file_sizes)
    if len(unique_sizes) == 1:
        print("All .pth files have the same size.")
    else:
        print("Not all .pth files have the same size:")

if __name__ == "__main__":
    folder_path = '/media/data/iter3/train'
    if os.path.isdir(folder_path):
        check_pth_file_sizes(folder_path)
    else:
        print("Invalid folder path. Please enter a valid directory.")
