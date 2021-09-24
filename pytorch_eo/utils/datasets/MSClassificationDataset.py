from .ClassificationDataset import ClassificationDataset
from pytorch_eo.utils import read_ms_image

class MSClassificationDataset(ClassificationDataset):
    def __init__(self, images, labels, trans, bands, norm_value):
        super().__init__(images, labels, trans, norm_value)

        # convert bands from enum to values
        if isinstance(bands, list):
            if len(bands) == 1:
                self.bands = bands[0].value
            else:
                self.bands = [band.value for band in bands]
        else:
            self.bands = bands.value

    def _read_image(self, img):
        return read_ms_image(img, self.bands)

    def _norm_image(self, img):
        return (img / self.norm_value).clip(0, 1)
