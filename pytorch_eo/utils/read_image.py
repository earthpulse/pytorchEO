import torchvision
import rasterio
from skimage import io
from einops import rearrange


def read_image(src):
    # return torchvision.io.read_image(src)  # C, H, W
    return io.imread(src)  # H, W, C


def read_ms_image(src, bands):
    img_data = rasterio.open(src).read(bands)
    if img_data.ndim == 3:
        return rearrange(img_data, 'c h w -> h w c')
    return img_data
