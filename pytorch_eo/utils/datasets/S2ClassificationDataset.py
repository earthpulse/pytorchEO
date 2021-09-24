from .ClassificationDataset import ClassificationDataset
from ..utils import read_ms_image
from ..sensors import bands2values

class S2ClassificationDataset(ClassificationDataset):
    def __init__(self, images, labels=None, trans=None, bands=None, norm_value=4000):
        super().__init__(images, labels, trans, norm_value)
        self.bands = bands2values(bands)

    def _read_image(self, img):
        return read_ms_image(img, self.bands)

    def _norm_image(self, img):
        return (img / self.norm_value).clip(0, 1)