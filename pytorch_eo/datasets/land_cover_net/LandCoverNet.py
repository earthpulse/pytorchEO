from pytorch_eo.utils import untar_file
import pytorch_lightning as pl
import os
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_eo.utils.datasets.SBSegmentationDataset import SBSegmentationDataset
import pandas as pd
import numpy as np
import torch


class Dataset(SBSegmentationDataset):

    def __init__(self, images, masks=None, trans=None, bands=('B04', 'B03', 'B02'), num_classes=8, norm_value=4000):
        super().__init__(images, masks, trans, bands, num_classes, norm_value)

    def _read_mask(self, mask):
        return super()._read_mask(mask)[..., 0]


class LandCoverNet(pl.LightningDataModule):

    # THIS DATASET NEEDS TO BE DOWNLOADED THROUGH https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/
    # CAN WE HAVE A PUBLIC LINK ?

    # WE HAVE 1980 LABELS, DERIVED FROM S2 TIME SERIES
    # THIS DATASET ASSIGNS THE SAME MASK TO ALL IMAGES IN THE SAME TIME SERIES

    def __init__(self,
                 batch_size,
                 path='/data',
                 compressed_data_filename='ref_landcovernet_v1_source.tar',
                 compressed_labels_filename='ref_landcovernet_v1_labels.tar',
                 data_folder='ref_landcovernet_v1_source',
                 labels_folder='ref_landcovernet_v1_labels',
                 test_size=0.2,
                 val_size=0.2,
                 random_state=42,
                 num_workers=0,
                 pin_memory=False,
                 shuffle=True,
                 verbose=True
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.compressed_data_filename = compressed_data_filename
        self.compressed_labels_filename = compressed_labels_filename
        self.num_classes = 10
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.verbose = verbose
        self.data_folder = data_folder
        self.labels_folder = labels_folder
        self.classes = [  # class position in the list corresponds to value in mask
            {'name': 'other', 'color': '#000000'},
            {'name': 'water', 'color': '#0000ff'},
            {'name': 'artificial-bare-ground', 'color': '#888888'},
            {'name': 'natural-bare-ground', 'color': '#d1a46d'},
            {'name': 'permanent-snow-ice', 'color': '#f5f5ff'},
            {'name': 'cultivated-vegetation', 'color': '#d64c2b'},
            {'name': 'permanent-snow-and-ice', 'color': '#186818'},
            {'name': 'semi-natural-vegetation', 'color': '#00ff00'},
        ]
        self.num_classes = len(self.classes)
        self.in_chans = 3

    def uncompress(self, compressed_data_filename, data_folder, msg):

        compressed_data_path = self.path / compressed_data_filename
        uncompressed_data_path = self.path / data_folder
        if not os.path.isdir(uncompressed_data_path):
            untar_file(compressed_data_path, self.path, msg=msg)
        else:  # TODO: Validate data is correct
            pass

    def setup(self, stage=None):

        # self.uncompress(self.compressed_labels_filename,
        #                 self.labels_folder, 'extracting labels ...')
        # # muy lento
        # self.uncompress(self.compressed_data_filename,
        #                 self.data_folder, 'extracting images ...')

        # generate list of images and labels
        labels = glob.glob(
            f'{self.path / self.labels_folder / self.labels_folder}*')

        # load dates
        images, masks = [], []
        for label in labels:
            source_dates = pd.read_csv(f'{label}/source_dates.csv')
            tile_id = label.split('_')[-2]
            chip_id = label.split('_')[-1]
            dates = source_dates[tile_id].values
            images += [f'{self.path}/{self.data_folder}/{self.data_folder}_{tile_id}_{chip_id}_{date}' for date in dates]
            masks += [f'{label}/labels.tif']*len(dates)

        # check images exist
        for image, mask in zip(images, masks):
            assert os.path.isdir(image), f'image {image} not found'
            assert os.path.isfile(mask), f'mask {mask} not found'

        # data splits (can we stratify ?)
        test_size = int(len(images) * self.test_size)
        val_size = int(len(images) * self.val_size)

        train_images, self.test_images, train_masks, self.test_masks = train_test_split(
            images,
            masks,
            test_size=test_size,
            random_state=self.random_state
        )

        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(
            train_images,
            train_masks,
            test_size=val_size,
            random_state=self.random_state
        )

        if self.verbose:
            print("training samples", len(self.train_images))
            print("validation samples", len(self.val_images))
            print("test samples", len(self.test_images))

        # datasets

        self.train_ds = Dataset(
            self.train_images, self.train_masks, num_classes=self.num_classes)
        self.val_ds = Dataset(
            self.val_images, self.val_masks, num_classes=self.num_classes)
        self.test_ds = Dataset(
            self.test_images, self.test_masks, num_classes=self.num_classes)

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
