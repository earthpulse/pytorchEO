from .EuroSATBase import EuroSATBase
from pytorch_eo.utils.datasets.MSClassificationDataset import MSClassificationDataset


class EuroSAT(EuroSATBase):

    def __init__(self,
                 batch_size,
                 download=True,
                 path="./data",
                 test_size=0.2,
                 val_size=0.2,
                 random_state=42,
                 num_workers=0,
                 pin_memory=False,
                 shuffle=True,
                 bands=None,
                 verbose=True
                 ):

        url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
        compressed_data_filename = 'EuroSATallBands.zip'
        data_folder = 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        super().__init__(batch_size, download, url, path,
                         compressed_data_filename, data_folder, test_size, val_size, random_state,
                         num_workers, pin_memory, shuffle, verbose)
        self.bands = bands
        self.in_chans = len(bands)

    def setup(self, stage=None):
        super().setup(stage=stage)

        self.train_ds = MSClassificationDataset(
            self.train_images, self.train_labels, bands=self.bands)

        self.val_ds = MSClassificationDataset(
            self.val_images, self.val_labels, bands=self.bands)

        self.test_ds = MSClassificationDataset(
            self.test_images, self.test_labels, bands=self.bands)
