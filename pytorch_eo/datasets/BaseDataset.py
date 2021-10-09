import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split


class BaseDataset(pl.LightningDataModule):
    def __init__(self, batch_size, test_size, val_size, verbose, num_workers, pin_memory, seed):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.verbose = verbose
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        # download, process, generate lists of samples
        # create dataset -> self.ds should be defined here
        pass

    def make_splits(self, stratify=None):

        if self.test_size:
            test_len = int(len(self.df)*self.test_size)
            train_df, self.test_df = train_test_split(
                self.df,
                test_size=test_len,
                random_state=self.seed,
                stratify=self.df[stratify] if stratify else None
            )
        else:
            train_df = self.df

        if self.val_size:
            val_len = int(len(self.df)*self.val_size)
            self.train_df, self.val_df = train_test_split(
                train_df,
                test_size=val_len,
                random_state=self.seed,
                stratify=train_df[stratify] if stratify else None
            )
        else:
            self.train_df = train_df

        if self.verbose:
            print("training samples", len(self.train_df))
            if self.val_size:
                print("validation samples", len(self.val_df))
            if self.test_size:
                print("test samples", len(self.test_df))

    def build_datasets(self):
        self.train_ds = self.build_dataset(self.train_df, self.train_trans)
        self.val_ds, self.test_ds = None, None
        if self.test_size:
            self.test_ds = self.build_dataset(self.test_df, self.test_trans)
        if self.val_size:
            self.val_ds = self.build_dataset(self.val_df, self.val_trans)

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
