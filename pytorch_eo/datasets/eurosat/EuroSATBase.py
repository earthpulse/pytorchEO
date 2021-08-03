from pytorch_eo.utils import download_url, unzip_file
import pytorch_lightning as pl
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader


class EuroSATBase(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 download,
                 url,
                 path,
                 compressed_data_filename,
                 data_folder,
                 test_size,
                 val_size,
                 random_state,
                 num_workers,
                 pin_memory,
                 shuffle,
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
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.verbose = verbose

    def setup(self, stage=None):

        # download data
        compressed_data_path = self.path / self.compressed_data_filename
        uncompressed_data_path = self.path / self.data_folder
        if self.download:
            # create data folder
            os.makedirs(self.path, exist_ok=True)
            # check data is not already downloaded
            if not os.path.isfile(compressed_data_path):
                print("downloading data ...")
                download_url(self.url, compressed_data_path)
            else:
                print("data already downloaded !")
            # extract
            if not os.path.isdir(uncompressed_data_path):
                unzip_file(compressed_data_path, self.path,
                           msg="extracting data ...")
            else:
                print("data already extracted !")
                # TODO: check data is correct
        else:
            assert os.path.isdir(uncompressed_data_path), 'data not found'
            # TODO: check data is correct

        # retrieve classes from folder structure
        self.classes = sorted(os.listdir(uncompressed_data_path))
        assert len(self.classes) == self.num_classes

        # generate list of images and labels
        images, encoded = [], []
        for ix, label in enumerate(self.classes):
            _images = os.listdir(uncompressed_data_path / label)
            images += [uncompressed_data_path /
                       label / img for img in _images]
            encoded += [ix]*len(_images)
        if self.verbose:
            print(f'Number of images: {len(images)}')

        # data splits

        train_images, self.test_images, train_labels, self.test_labels = train_test_split(
            images,
            encoded,
            stratify=encoded,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(
            train_images,
            train_labels,
            stratify=train_labels,
            test_size=self.val_size,
            random_state=self.random_state
        )

        if self.verbose:
            print("training samples", len(self.train_images))
            print("validation samples", len(self.val_images))
            print("test samples", len(self.test_images))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
