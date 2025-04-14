# PFAS Machine Learning Research Repository
## Introduction 

This repository is dedicated to preparing geographical data for machine learning applications, with a focus on convolutional neural networks (CNNs) and U-Net architectures. Its primary objective is to convert raw geospatial data into tensors, making them ready for use in training and validation workflows.

## Workflow
The data processing pipeline consists of several stages:

1. CSV to TIFF Conversion

    - Raw data points, provided in CSV format, are converted into TIFF files.
    - Convert coordinate system to correct standard.
    - These TIFFs are cropped to align with the ground truth data, ensuring a one-to-one correspondence between the input data and target labels.

2. Processing Existing TIFFs - preprocessing

    - Pre-existing TIFF files are resized and cropped to match the target format.
    - Convert coordinate system to correct standard.
    - The coordinate reference system (CRS) is standardized to ensure consistency with the ground truth data.

3. TIFF to Tensor Conversion

    - TIFF files are sampled into 32x32 patches.
    - These patches are converted into PyTorch tensors and saved as .pth files, maintaining alignment with corresponding ground truth data.
    
4. Model Training and Validation

- A custom dataloader (src/dataloader.py) is used to efficiently load the processed tensor data into machine learning models.
- An example training and validation pipeline is provided in src/train.py, demonstrating how to use the prepared data for model development.

This streamlined process ensures that geographical data is correctly formatted and ready for machine learning workflows, facilitating the development of accurate predictive models.
## Set Up Environment

- Install python 3.10.14. (we used pyenv to instantiate our virtual environment [pyenv](https://realpython.com/intro-to-pyenv/))
- run `pip install -r requirements.txt` (there might be issues with the torch version and the specific GPU driver)

## Processing for csvs

- Rename longitude and latitude columns in every csv to lon and lat respectively.
- Divide csvs into discrete and categorical data.

Rename the target value in the discrete data to 'target' and create a new column named target in the categorical data and assign it the value of 1 for all rows. 
(remember to normalize values)

Refer to readme in src/ folder for commands

## Processing existing rasters (tiffs)

Check out the preprocess folder. There several tools to help check the consistency and other metadata of rasters. In addition, we also want to standardize how null values are handled. These scripts are used before final processing into tensors to ensure consistency. It is also helpful to plot the rasters with matplotlib, please check out the `create_plot.py` and the notebooks in the `notebook_tests` folder.

Refer to readme in src/ folder for commands

## Process raster into tensors

The assumption here is that all rasters are in a standard format in terms of resolution, pixel size, etc. We then take invididual grid cells from the raster and map them to the ground truth. Please read through `tiff_to_tensor.py` for details. 

Refer to readme in src/ folder for command details

## data augment - WIP

## Notes / Troubleshooting

TBD

## TODOS

- data augmentation = flips and whatever we do to images
    - random resize?
    - random resize crop?
    - random rotate
    - gaussian blur/noise
- config pixel size = think more on this
- change the way we handle null vals -50 -> 0 -> 1
- maybe change cell stacking from (x, 1, 10, 10) -> (x, 10, 10)
