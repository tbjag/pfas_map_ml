import torch
import os
from pathlib import Path

# Directory containing the .pth files
input_dir = '/media/data/iter3/img_target/v3'
output_dir = '/media/data/iter3/img_target/v3_scaled'  # Directory to save upscaled files

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Scaling factor
scale_factor = 20000

# Process each .pth file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.pth'):
        # Load the tensor
        filepath = os.path.join(input_dir, filename)
        tensor = torch.load(filepath)
        
        # Verify the shape is 3x32x32
        if tensor.shape == (3, 32, 32):
            # Scale the tensor
            scaled_tensor = tensor * scale_factor
            
            # Save the scaled tensor
            output_path = os.path.join(output_dir, filename)
            torch.save(scaled_tensor, output_path)
            print(f"Processed: {filename}")
        else:
            print(f"Skipped {filename} - unexpected shape {tensor.shape}")

print("All files processed!")