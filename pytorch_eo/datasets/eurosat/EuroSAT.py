from .EuroSATBase import EuroSATBase
from pytorch_eo.utils.datasets.S2ClassificationDataset import S2ClassificationDataset
from ...utils.sensors import S2

class EuroSAT(EuroSATBase):

    def __init__(self,
                 batch_size,
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
                 trans=None,
                 dataset=None,
                 norm_value=4000
                 ):

        url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
        compressed_data_filename = 'EuroSATallBands.zip'
        data_folder = 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        super().__init__(batch_size, download, url, path,
                         compressed_data_filename, data_folder, 
                         train_sampler, test_sampler, val_sampler,
                         test_size, val_size, 
                         num_workers, pin_memory, seed, verbose)
        
        self.bands = bands
        if bands is None:
            self.bands = S2.ALL

        if isinstance(self.bands, list):
            self.in_chans = len(bands)
            for band in self.bands:
                assert band in S2, 'invalid band'
        else:
            assert self.bands in S2, 'invalid band'
            if isinstance(self.bands.value, list):
                self.in_chans = len(self.bands.value)
            else:
                self.in_chans = 1            

        self.trans = trans
        self.dataset = dataset
        self.norm_value = norm_value

    def build_dataset(self):
        if self.dataset:
            return self.dataset(self.images, self.labels, self.trans, self.bands, self.norm_value)
        return S2ClassificationDataset(self.images, self.labels, self.trans, self.bands, self.norm_value)

