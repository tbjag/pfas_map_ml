import numpy as np
import os

def generate_random_indices(length, output_file='random_indices.txt'):
    """
    Generate a random permutation of indices and save to disk
    
    Parameters:
    length (int): The length of the sequence to generate indices for
    output_file (str): Path to save the indices (default: 'random_indices.txt')
    
    Returns:
    numpy.ndarray: The generated random indices
    """
    # Generate random permutation
    indices = np.random.permutation(length)
    
    # Save to disk
    np.savetxt(output_file, indices, fmt='%d')
    
    print(f"Random indices saved to {output_file}")
    return indices

def load_random_indices(input_file='random_indices.txt'):
    """
    Load previously saved random indices
    
    Parameters:
    input_file (str): Path to the saved indices file
    
    Returns:
    numpy.ndarray: The loaded indices
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    
    indices = np.loadtxt(input_file, dtype=int)
    return indices