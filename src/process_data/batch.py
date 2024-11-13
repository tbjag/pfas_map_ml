import torch
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

class PTHBatchProcessor:
    def __init__(self, input_dir, output_dir, batch_size=1000, matrix_size=(10, 10)):
        """
        Initialize the batch processor for .pth files
        
        Args:
            input_dir (str): Directory containing input .pth files
            output_dir (str): Directory to save processed batches
            batch_size (int): Number of samples per batch
            matrix_size (tuple): Size of the matrix in each sample (default: (10, 10))
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.matrix_size = matrix_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_pth_files(self):
        """Get list of all .pth files in input directory"""
        return sorted(self.input_dir.glob("*.pth"))
    
    def load_and_process_batch(self, file_paths, start_idx, end_idx):
        """
        Load a specific range of indices from multiple .pth files and combine them
        
        Args:
            file_paths (list): List of paths to .pth files
            start_idx (int): Start index for the batch
            end_idx (int): End index for the batch
            
        Returns:
            torch.Tensor: Combined batch tensor
        """
        batch_size = end_idx - start_idx
        num_files = len(file_paths)
        
        # Initialize batch tensor
        batch_data = torch.zeros((batch_size, num_files, *self.matrix_size))
        
        # Load relevant slice from each file
        for file_idx, file_path in enumerate(file_paths):
            try:
                # Load full tensor
                data = torch.load(file_path)
                # Extract relevant slice
                batch_data[:, file_idx] = data[start_idx:end_idx, 0]
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return None
                
        return batch_data
    
    def process_files(self):
        """Process all files in batches and save to output directory"""
        pth_files = self.get_pth_files()
        if not pth_files:
            print("No .pth files found in input directory")
            return
            
        # Load first file to get total number of samples
        first_file = torch.load(pth_files[0])
        total_samples = first_file.shape[0]
        
        # Calculate number of batches
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {len(pth_files)} files with {total_samples} samples in {num_batches} batches")
        
        # Process each batch
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_samples)
            
            # Load and process batch
            batch_data = self.load_and_process_batch(pth_files, start_idx, end_idx)
            
            if batch_data is not None:
                # Save batch
                output_path = self.output_dir / f"batch_{batch_idx:05d}.pth"
                torch.save(batch_data, output_path)
                
    def load_batch(self, batch_idx):
        """
        Load a specific batch for training/evaluation
        
        Args:
            batch_idx (int): Index of the batch to load
            
        Returns:
            torch.Tensor: Batch tensor
        """
        batch_path = self.output_dir / f"batch_{batch_idx:05d}.pth"
        if batch_path.exists():
            return torch.load(batch_path)
        return None

# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "../../data/pth_temp"
    OUTPUT_DIR = "../../data/pth_batch"
    BATCH_SIZE = 1000  # Adjust based on your RAM constraints
    
    # Initialize and run processor
    processor = PTHBatchProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Process all files into batches
    processor.process_files()
    
    # Example of loading a batch for training
    batch = processor.load_batch(0)
    if batch is not None:
        print(f"Loaded batch shape: {batch.shape}")