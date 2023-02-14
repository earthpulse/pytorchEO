from .SensorImageDataset import SensorImageDataset
from pytorch_eo.utils import read_sb_image
from pytorch_eo.datasets.sensors import bands2names
from einops import rearrange
import numpy as np


class SingleBandImageDataset(SensorImageDataset):
    def __init__(self, images, sensor, bands, prefix=None, ext=".tif", lowercase=False):
        super().__init__(images, sensor, bands)
        self.prefix = prefix
        self.bands = bands2names(bands)
        self.lowercase = lowercase
        self.ext = ext

    def __getitem__(self, ix):
        prefix = self.prefix[ix] if self.prefix is not None else None
        img = read_sb_image(
            self.images[ix], self.bands, prefix, self.ext, self.lowercase
        )
        if img.dtype == np.uint16:  # uin16 is not supported by pytorch
            img = img.astype(np.float32)
        if img.ndim == 3:
            return rearrange(img, "c h w -> h w c")
        return img
