from .BaseDataset import BaseDataset
from ..utils.sensors import S2


class S2Dataset(BaseDataset):
    def __init__(self, batch_size, train_sampler, test_sampler, val_sampler, test_size, val_size, verbose, num_workers, pin_memory, seed, bands):
        super().__init__(batch_size, train_sampler, test_sampler, val_sampler,
                         test_size, val_size, verbose, num_workers, pin_memory, seed)

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
