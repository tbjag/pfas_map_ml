# PFAS Machine Learning Research Repository
## Introduction
This repository focuses on preparing geographical data for machine learning models, particularly convolutional neural networks (CNNs) and U-Net architectures. The goal is to transform raw geospatial data into tensors that can be directly used for training and validation.

## Workflow
The data processing pipeline consists of several stages:

1. CSV to TIFF Conversion

    - Raw data points, provided in CSV format, are converted into TIFF files.
    - These TIFFs are cropped to align with the ground truth data, ensuring a one-to-one correspondence between the input data and target labels.

2. Processing Existing TIFFs

    - Pre-existing TIFF files are resized and cropped to match the target format.
    - The coordinate reference system (CRS) is standardized to ensure consistency with the ground truth data.

3. TIFF to Tensor Conversion

    - TIFF files are sampled into 32x32 patches.
    - These patches are converted into PyTorch tensors and saved as .pth files, maintaining alignment with corresponding ground truth data.
    
4. Model Training and Validation

- A custom dataloader (src/dataloader.py) is used to efficiently load the processed tensor data into machine learning models.
- An example training and validation pipeline is provided in src/train.py, demonstrating how to use the prepared data for model development.

This streamlined process ensures that geographical data is correctly formatted and ready for machine learning workflows, facilitating the development of accurate predictive models.
## Set Up Environment

- Install python 3.10.14. (we used pyenv to instantiate our virtual environment (insert link)
- run `pip install -r requirements.txt` (there might be issues with the torch version and the specific GPU driver)

## WIP below

## Notes / Troubleshooting

python csv_to_tif.py 

## TODOS

- change dataloader to add in target
- test dataloader on unet

- data augmentation = flips and whatever we do to images
- add readme to processing file folder
- add error checking for lat/lon/target inputs
- for target = specify discrete or boolean
- config pixel size = think more on this
- change the way we handle null vals -50 -> 0 -> 1
- maybe change cell stacking from (x, 1, 10, 10) -> (x, 10, 10)
