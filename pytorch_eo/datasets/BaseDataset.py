import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split, SequentialSampler, SubsetRandomSampler
import torch


class BaseDataset(pl.LightningDataModule):
    def __init__(self, batch_size, train_sampler, test_sampler, val_sampler, test_size, val_size, verbose, num_workers, pin_memory, seed):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.test_size = test_size
        self.val_size = val_size
        self.verbose = verbose
        self.seed = seed

    def setup(self, stage=None):
        # download, process, generate lists of samples
        # create dataset -> self.ds should be defined here
        pass

    def make_splits(self):
        if not self.train_sampler:
            # data splits (random splits)
            idxs = list(range(len(self.ds)))
            testset_len = int(len(self.ds)*self.test_size)
            trainset_len = len(self.ds) - testset_len
            train_idxs, self.test_idxs = random_split(
                idxs, [trainset_len, testset_len], generator=torch.Generator().manual_seed(self.seed))
            valset_len = int(len(self.ds)*self.val_size)
            trainset_len = trainset_len - valset_len
            self.train_idxs, self.val_idxs = random_split(train_idxs, [
                                                          trainset_len, valset_len], generator=torch.Generator().manual_seed(self.seed))

            self.train_sampler = SubsetRandomSampler(self.train_idxs)
            self.val_sampler = SequentialSampler(self.val_idxs)
            self.test_sampler = SequentialSampler(self.test_idxs)

            if not self.train_sampler:
                raise ValueError("train sampler should be definied")

        if self.verbose:
            print("training samples", len(self.train_sampler))
            if self.val_sampler:
                print("validation samples", len(self.val_sampler))
            if self.test_sampler:
                print("test samples", len(self.test_sampler))

    def get_dataloader(self, sampler):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler
        )
    # train_dataloader is required, the others are optional (but recommended!)

    def train_dataloader(self):
        return self.get_dataloader(self.train_sampler)

    def val_dataloader(self):
        return self.get_dataloader(self.val_sampler) if self.val_sampler else None

    def test_dataloader(self):
        return self.get_dataloader(self.test_sampler) if self.test_sampler else None
