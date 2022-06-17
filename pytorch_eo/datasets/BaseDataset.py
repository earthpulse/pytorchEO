import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch


class BaseDataset(pl.LightningDataModule):
    def __init__(self, batch_size=16, verbose=True, num_workers=0, pin_memory=False, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        raise NotImplementedError()

    def build_datasets(self, train, test = None, val = None):
        self.train_ds = self.build_dataset(train)
        self.val_ds, self.test_ds = None, None
        if test is not None:
            self.test_ds = self.build_dataset(test)
        if val is not None:
            self.val_ds = self.build_dataset(val)

    def get_dataloader(self, ds, shuffle=False, batch_size=None):
        return DataLoader(
            ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle
        )

    # train_dataloader is required, the others are optional (but recommended!)

    def train_dataloader(self, shuffle=True, batch_size=None):
        return self.get_dataloader(self.train_ds, shuffle, batch_size=batch_size)

    def val_dataloader(self, shuffle=False, batch_size=None):
        return self.get_dataloader(self.val_ds, shuffle, batch_size=batch_size) if self.val_ds is not None else None

    def test_dataloader(self, shuffle=False, batch_size=None):
        return self.get_dataloader(self.test_ds, shuffle, batch_size=batch_size) if self.test_ds is not None else None
