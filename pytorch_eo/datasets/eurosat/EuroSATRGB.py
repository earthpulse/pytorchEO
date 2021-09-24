from pytorch_eo.utils.datasets.ClassificationDataset import ClassificationDataset
from .EuroSATBase import EuroSATBase


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
                 trans=None,
                 dataset=None
                 ):

        url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        compressed_data_filename = 'EuroSAT.zip'
        data_folder = '2750'
        super().__init__(batch_size, download, url, path,
                         compressed_data_filename, data_folder, 
                         train_sampler, test_sampler, val_sampler,
                         test_size, val_size, 
                         num_workers, pin_memory, seed, verbose)
        self.in_chans = 3
        self.trans = trans
        self.dataset = dataset

    def build_dataset(self):
        if self.dataset:
            return self.dataset(self.images, self.labels, self.trans)
        return ClassificationDataset(self.images, self.labels, self.trans)


       
