import os
import rasterio

dir = '/media/data/iter3/temp'
for filename in os.listdir(dir):
    filepath = os.path.join(dir, filename)
    with rasterio.open(filepath) as dataset:
        print(f'{filename}; w: {dataset.width}; h: {dataset.height}')

