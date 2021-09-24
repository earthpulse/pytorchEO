from .MSClassificationDataset import MSClassificationDataset

class S2ClassificationDataset(MSClassificationDataset):
    def __init__(self, images, labels=None, trans=None, bands=None, norm_value=4000):
        super().__init__(images, labels, trans, bands, norm_value)
