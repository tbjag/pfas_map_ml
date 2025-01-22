import torch
import torchvision.transforms.v2 as transforms
from pathlib import Path
from tqdm import tqdm
import argparse
import os

class AddGaussianNoise:
    """Custom transform to add Gaussian noise"""
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

def get_highest_file_number(directory):
    """Get the highest numbered file in the directory"""
    numbers = []
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            try:
                numbers.append(int(filename.split('.')[0]))
            except ValueError:
                continue
    return max(numbers) if numbers else -1

def augment_dataset(train_dir, target_dir):
    """
    Augment dataset using transform pipelines with safe file numbering.

    Args:
        train_dir: Directory containing training data (/train)
        target_dir: Directory containing target data (/target)
    """
    # Get the highest existing file number
    highest_number = get_highest_file_number(train_dir)
    next_file_number = highest_number + 1

    # Define transform pipelines
    geometric_pipelines = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0)
        ]),
        transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0)
        ]),
        transforms.Compose([
            transforms.RandomRotation(degrees=(90, 90))
        ]),
        transforms.Compose([
            transforms.RandomRotation(degrees=(180, 180))
        ]),
        transforms.Compose([
            transforms.RandomRotation(degrees=(270, 270))
        ])
    ]

    data_only_pipelines = [
        transforms.Compose([
            AddGaussianNoise(std=0.05)
        ]),
        transforms.Compose([
            AddGaussianNoise(std=0.1)
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(
                size=(32, 32),
                scale=(0.8, 0.9),
                ratio=(1.0, 1.0)
            )
        ])
    ]

    original_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.pth')])

    print(f"Found {len(original_files)} original files")
    print(f"Starting new files from number: {next_file_number}")

    total_augmentations = len(geometric_pipelines) + len(data_only_pipelines)
    total_files_to_create = len(original_files) * total_augmentations
    print(f"Will create {total_files_to_create} new files")

    files_created = 0
    with tqdm(total=total_files_to_create, desc="Augmenting", unit="file") as pbar:
        for filename in original_files:
            data_path = os.path.join(train_dir, filename)
            target_path = os.path.join(target_dir, filename)

            data = torch.load(data_path)
            target = torch.load(target_path)

            for pipeline in geometric_pipelines:
                transformed_data = pipeline(data)
                transformed_target = pipeline(target)

                new_filename = f"{next_file_number:07d}.pth"
                next_file_number += 1

                torch.save(transformed_data, os.path.join(train_dir, new_filename))
                torch.save(transformed_target, os.path.join(target_dir, new_filename))

                files_created += 1
                pbar.update(1)

            for pipeline in data_only_pipelines:
                transformed_data = pipeline(data)

                new_filename = f"{next_file_number:07d}.pth"
                next_file_number += 1

                torch.save(transformed_data, os.path.join(train_dir, new_filename))
                torch.save(target, os.path.join(target_dir, new_filename))

                files_created += 1
                pbar.update(1)

    print(f"Augmentation complete. Created {files_created} new files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a dataset with various transformations.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target data directory.")

    args = parser.parse_args()

    augment_dataset(args.train_dir, args.target_dir)
