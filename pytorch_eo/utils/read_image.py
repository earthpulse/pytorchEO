import torchvision
import torch
import rasterio
import numpy as np

def read_image(src):
    return torchvision.io.read_image(src) # C, H, W
    
# this is very slow compared to reading the full image and keep the bands with indices

def read_ms_image(src, bands):
    img_data = rasterio.open(src).read(bands) # C, H, W
    # ms image can be uin16, but torch does not like it
    return torch.from_numpy(img_data.astype(np.float32))