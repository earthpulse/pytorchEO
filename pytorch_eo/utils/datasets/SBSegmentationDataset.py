import torch

from .SegmentationDataset import SegmentationDataset
from ..read_image import read_ms_image
from ..sensors import bands2names


class SBSegmentationDataset(SegmentationDataset):
    def __init__(self, images, masks=None, trans=None, bands=None, num_classes=None, norm_value=4000):
        super().__init__(images, masks, trans, num_classes, norm_value)
        self.bands = bands2names(bands)

    def _read_image(self, img):
        bands_data = []
        for band in self.bands:
            band = read_ms_image(f'{img}/{band}.tif', 1)  # 1, H, W
            bands_data.append(band)
        img = torch.stack(bands_data).squeeze(1)  # C, H, W
        return img

    def _norm_image(self, img):
        return (img / self.norm_value).clip(0, 1)
