from .S2ImageDataset import S2ImageDataset
from ..read_image import read_sb_image
from ..sensors import bands2names
from einops import rearrange
import numpy as np


class S2SBImageDataset(S2ImageDataset):
    def __init__(self, images, bands):
        super().__init__(images, bands)
        self.bands = bands2names(bands)

    def __getitem__(self, ix):
        img = read_sb_image(self.images[ix], self.bands)
        # uin16 is not supported by pytorch
        img = img.astype(np.float32)
        return rearrange(img, 'c h w -> h w c')
