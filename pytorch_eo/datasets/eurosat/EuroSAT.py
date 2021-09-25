from .EuroSATBase import EuroSATBase
from ...utils.datasets.S2ImageDataset import S2ImageDataset


class EuroSAT(EuroSATBase):

    def __init__(self,
                 batch_size=32,
                 download=True,
                 path="./data",
                 train_sampler=None,
                 test_sampler=None,
                 val_sampler=None,
                 test_size=0.2,
                 val_size=0.2,
                 num_workers=0,
                 pin_memory=False,
                 seed=42,
                 verbose=False,
                 bands=None,
                 norm_value=4000
                 ):
        super().__init__(batch_size, download, path, train_sampler, test_sampler,
                         val_sampler, test_size, val_size, num_workers, pin_memory, seed, verbose)
        self.url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
        self.compressed_data_filename = 'EuroSATallBands.zip'
        self.data_folder = 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        self.bands = bands
        self.norm_value = norm_value

    def get_image_ds(self, images):
        return S2ImageDataset(images, self.bands, self.norm_value)
