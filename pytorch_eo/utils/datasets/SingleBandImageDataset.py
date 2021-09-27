from .SensorImageDataset import SensorImageDataset
from ..read_image import read_sb_image
from ..sensors.utils import bands2names
from einops import rearrange
import numpy as np


class SingleBandImageDataset(SensorImageDataset):
    def __init__(self, images, sensor, bands, prefix=None):
        super().__init__(images, sensor, bands)
        self.prefix=prefix
        self.bands = bands2names(bands)

    def __getitem__(self, ix):
        prefix = self.prefix[ix] if self.prefix is not None else None
        img = read_sb_image(self.images[ix], self.bands, prefix)
        if img.dtype == np.uint16: # uin16 is not supported by pytorch
            img = img.astype(np.float32)
        if img.ndim == 3:
            return rearrange(img, 'c h w -> h w c')
        return img
