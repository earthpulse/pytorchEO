from pytorch_eo.utils import download_url, unzip_file
import pytorch_lightning as pl
import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split, SequentialSampler, SubsetRandomSampler
import torch

class EuroSATBase(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 download,
                 url,
                 path,
                 compressed_data_filename,
                 data_folder,
                 train_sampler,
                 test_sampler,
                 val_sampler,
                 test_size,
                 val_size,
                 num_workers,
                 pin_memory,
                 seed,
                 verbose
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.download = download
        self.path = Path(path)
        self.url = url
        self.compressed_data_filename = compressed_data_filename
        self.data_folder = data_folder
        self.num_classes = 10
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.test_size = test_size 
        self.val_size = val_size
        self.seed = seed

    def build_dataset(self):
        print("this function must be defined!")
        pass

    def setup(self, stage=None):

        self.download_data()
        self.generate_classes_list()

        # generate list of images and labels
        self.images, self.labels = [], []
        for ix, label in enumerate(self.classes):
            _images = os.listdir(self.uncompressed_data_path / label)
            self.images += [str(self.uncompressed_data_path /
                       label / img) for img in _images]
            self.labels += [ix]*len(_images)
        assert len(self.images) == len(self.labels)
        if self.verbose:
            print(f'Number of images: {len(self.images)}')

        self.ds = self.build_dataset()

        if not self.train_sampler:
            # data splits (random splits)
            idxs = list(range(len(self.ds)))
            testset_len = int(len(self.ds)*self.test_size)
            trainset_len = len(self.ds) - testset_len
            train_idxs, self.test_idxs = random_split(idxs, [trainset_len, testset_len], generator=torch.Generator().manual_seed(self.seed))
            valset_len = int(len(self.ds)*self.val_size)
            trainset_len = trainset_len - valset_len
            self.train_idxs, self.val_idxs = random_split(train_idxs, [trainset_len, valset_len], generator=torch.Generator().manual_seed(self.seed))

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
    
    def download_data(self):
        # download data
        compressed_data_path = self.path / self.compressed_data_filename
        self.uncompressed_data_path = self.path / self.data_folder
        if self.download:
            # create data folder
            os.makedirs(self.path, exist_ok=True)

            # extract
            if not os.path.isdir(self.uncompressed_data_path):

                # check data is not already downloaded
                if not os.path.isfile(compressed_data_path):
                    print("downloading data ...")
                    download_url(self.url, compressed_data_path)
                else:
                    print("data already downloaded !")

                unzip_file(compressed_data_path, self.path,
                           msg="extracting data ...")
            else:
                if self.verbose:
                    print("data already extracted !")
                # TODO: check data is correct
        else:
            assert os.path.isdir(self.uncompressed_data_path), 'data not found'
            # TODO: check data is correct

    def generate_classes_list(self):
        # retrieve classes from folder structure
        self.classes = sorted(os.listdir(self.uncompressed_data_path))
        assert len(self.classes) == self.num_classes

    # train_dataloader is required, the others are optional (but recommended!)

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
                self.ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=self.val_sampler
            ) if self.val_sampler else None

    def test_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.test_sampler
        ) if self.test_sampler else None
