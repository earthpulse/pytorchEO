from pytorch_eo.utils.datasets.RGBClassificationDataset import RGBClassificationDataset
from .EuroSATBase import EuroSATBase


class EuroSATRGB(EuroSATBase):

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
                 verbose=True
                 ):

        url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        compressed_data_filename = 'EuroSAT.zip'
        data_folder = '2750'
        super().__init__(batch_size, download, url, path,
                         compressed_data_filename, data_folder, test_size, val_size, random_state,
                         num_workers, pin_memory, shuffle, verbose)
        self.in_chans = 3

    def setup(self, stage=None):
        super().setup(stage=stage)

        self.train_ds = RGBClassificationDataset(
            self.train_images, self.train_labels)

        self.val_ds = RGBClassificationDataset(
            self.val_images, self.val_labels)

        self.test_ds = RGBClassificationDataset(
            self.test_images, self.test_labels)
