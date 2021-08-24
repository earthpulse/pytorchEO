from .SegmentationDataset import SegmentationDataset
from pytorch_eo.utils import read_image
import torch
import numpy as np


class SBSegmentationDataset(SegmentationDataset):
    def __init__(self, images, masks=None, trans=None, bands=('B04', 'B03', 'B02'), num_classes=None, norm_value=4000):
        super().__init__(images, masks, trans, num_classes, norm_value)
        self.bands = bands

    def _read_image(self, img):
        bands_data = []
        for band in self.bands:
            band = read_image(f'{img}/{band}.tif')  # H, W
            bands_data.append(band)
        img = np.stack(bands_data)  # C, H, W
        return img

    def _img_to_tensor(self, img):
        return torch.from_numpy(img).float().clip(0, 1)  # C, H, W
