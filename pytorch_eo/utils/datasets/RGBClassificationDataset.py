from .ClassificationDataset import ClassificationDataset
from pytorch_eo.utils import read_image


class RGBClassificationDataset(ClassificationDataset):
    def __init__(self, images, labels=None, trans=None, norm_value=255):
        super().__init__(images, labels, trans, norm_value)

    def _read_image(self, img):
        return read_image(img)
