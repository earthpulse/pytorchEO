from .EuroSATBase import EuroSATBase
from ...utils.datasets.RGBImageDataset import RGBImageDataset


class EuroSATRGB(EuroSATBase):

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
                 trans=None
                 ):
        super().__init__(batch_size, download, path, train_sampler, test_sampler,
                         val_sampler, test_size, val_size, num_workers, pin_memory, seed, verbose, trans)
        self.url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        self.compressed_data_filename = 'EuroSAT.zip'
        self.data_folder = '2750'
        self.in_chans = 3

    def get_image_ds(self, images):
        return RGBImageDataset(images)
