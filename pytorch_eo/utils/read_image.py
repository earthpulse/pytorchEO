import rasterio
from skimage import io
from einops import rearrange
import numpy as np


def read_image(src):
    # return torchvision.io.read_image(src)  # C, H, W
    return io.imread(src)  # H, W, C


def read_ms_image(src, bands):
    img_data = rasterio.open(src).read(bands)
    if img_data.ndim == 3:
        return rearrange(img_data, 'c h w -> h w c')
    return img_data

# load tif image from folder with separated bands
# file name should be {BAND}.tif


def read_sb_image(src, bands):
    bands_data = []
    for band in bands:
        band = read_ms_image(f'{src}/{band}.tif', 1)  # H, W
        bands_data.append(band)
    return np.stack(bands_data)  # C, H, W
