import rasterio
from skimage import io
from einops import rearrange
import numpy as np


def read_image(src):
    return io.imread(src)  # H, W, C


def read_ms_image(src, bands):
    img_data = rasterio.open(src).read(bands)
    if img_data.ndim == 3:
        return rearrange(img_data, "c h w -> h w c")
    return img_data


# load tif image from folder with separated bands
# file name should be {BAND}.tif


def read_sb_image(src, bands, prefix=None, ext=".tif", lowercase=False):
    bands_data = []
    file_name = src + "/"
    if prefix is not None:
        file_name += prefix
    if len(bands) > 1:
        for band in bands:
            band = band.lower() if lowercase else band
            band = read_ms_image(file_name + band + ext, 1)  # H, W
            bands_data.append(band)
        return np.stack(bands_data)  # C, H, W
    return read_ms_image(file_name + bands[0] + ext, 1)  # H, W
