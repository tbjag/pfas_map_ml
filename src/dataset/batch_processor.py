import os
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional

class BatchProcessor:
    def __init__(
        self,
        data_dir: Union[str, Path],
        indices_file: str = 'random_indices.txt',
        batch_size: int = 32,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the batch processor
        
        Parameters:
        data_dir (str|Path): Directory containing H5 files
        indices_file (str): File containing random indices
        batch_size (int): Size of batches to process
        output_dir (str): Directory to save processed batches (defaults to data_dir/processed)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load indices
        self.indices = np.loadtxt(indices_file, dtype=int)
        
        # Get list of H5 files
        self.h5_files = sorted(list(self.data_dir.glob('*.h5')))
        if not self.h5_files:
            raise ValueError(f"No H5 files found in {self.data_dir}")

    def process_batches(self):
        """
        Process all data in batches and save as PyTorch files
        """
        num_batches = len(self.indices) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = self.indices[start_idx:end_idx]
            
            # Process this batch
            batch_data = self._process_batch(batch_indices)
            
            # Save batch
            output_path = self.output_dir / f"batch_{batch_idx:05d}.pt"
            torch.save(batch_data, output_path)
            
            print(f"Saved batch {batch_idx+1}/{num_batches} to {output_path}")

    def _process_batch(self, batch_indices: np.ndarray) -> torch.Tensor:
        """
        Process a single batch of indices
        
        Parameters:
        batch_indices (np.ndarray): Indices to process in this batch
        
        Returns:
        torch.Tensor: Concatenated data for this batch
        """
        batch_data = []
        
        for idx in batch_indices:
            # Determine which file and where in the file to read
            # This assumes the data is distributed across files sequentially
            file_idx = idx // self._samples_per_file()
            local_idx = idx % self._samples_per_file()
            
            if file_idx >= len(self.h5_files):
                raise IndexError(f"Index {idx} exceeds available data")
            
            # Read data from H5 file
            with h5py.File(self.h5_files[file_idx], 'r') as f:
                # Assuming the main dataset is named 'data'
                # Modify this according to your H5 file structure
                data = torch.from_numpy(f['data'][local_idx])
                batch_data.append(data)
        
        return torch.stack(batch_data)

    def _samples_per_file(self) -> int:
        """
        Get the number of samples in each H5 file
        Assumes all files have the same number of samples
        """
        with h5py.File(self.h5_files[0], 'r') as f:
            return len(f['cells'])