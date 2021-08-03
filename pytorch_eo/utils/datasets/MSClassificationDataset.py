from .ClassificationDataset import ClassificationDataset
from pytorch_eo.utils import read_ms_image


class MSClassificationDataset(ClassificationDataset):
    def __init__(self, images, labels=None, trans=None, bands=None, norm_value=4000):
        super().__init__(images, labels, trans, norm_value)
        self.bands = bands

    def _read_image(self, img):
        return read_ms_image(img, self.bands)

    def _norm_image(self, img):
        return (img / self.norm_value).clip(0, 1)
