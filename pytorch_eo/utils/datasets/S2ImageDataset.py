from .RasterioImageDataset import RasterioImageDataset
from ..sensors import S2, bands2values
import torch
import numpy as np

# load S2 images with rasterio


class S2ImageDataset(RasterioImageDataset):
    def __init__(self, images, bands):
        super().__init__(images, bands)

        # parse bands and compute number
        self.bands = bands
        if bands is None:
            self.bands = S2.ALL

        if isinstance(self.bands, list):
            self.in_chans = len(bands)
            for band in self.bands:
                assert band in S2, 'invalid band'
        else:
            assert self.bands in S2, 'invalid band'
            if isinstance(self.bands.value, list):
                self.in_chans = len(self.bands.value)
            else:
                self.in_chans = 1

        # convert to values
        self.bands = bands2values(self.bands)

    def __getitem__(self, ix):
        img_data = super().__getitem__(ix)
        # uin16 is not supported by pytorch
        return img_data.astype(np.float32)
