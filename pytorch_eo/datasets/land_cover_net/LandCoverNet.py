from pytorch_eo.utils import untar_file
import pytorch_lightning as pl
import os
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_eo.utils.datasets.SBSegmentationDataset import SBSegmentationDataset


class LandCoverNet(pl.LightningDataModule):

    # THIS DATASET NEEDS TO BE DOWNLOADED THROUGH https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/
    # CAN WE HAVE A PUBLIC LINK ?

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
        self.num_classes = 7
        self.classes = ['water',
                        'natural bare ground',
                        'artificial bare ground',
                        'woody vegetation',
                        'cultivated vegetation',
                        '(semi) natural vegetation',
                        'permanent snow/ice']

    def uncompress(self, compressed_data_filename, data_folder, msg):

        compressed_data_path = self.path / compressed_data_filename
        uncompressed_data_path = self.path / data_folder
        if not os.path.isdir(uncompressed_data_path):
            untar_file(compressed_data_path, self.path, msg=msg)
        else:  # TODO: Validate data is correct
            pass

    def setup(self, stage=None):

        self.uncompress(self.compressed_labels_filename,
                        self.labels_folder, 'extracting labels ...')
        # BUG: NOT WORKING ??? maybe demasiado lento y por eso no sale progress bar
        self.uncompress(self.compressed_data_filename,
                        self.data_folder, 'extracting images ...')

        # generate list of images and labels
        images = glob.glob(f'{self.path / self.data_folder}/*')
        # masks = glob.glob(f'{self.path / self.labels_folder}/*') # different number of files !
        masks = glob.glob(f'{self.path / self.data_folder}/*')

        # data splits (can we stratify ?)

        train_images, self.test_images, train_masks, self.test_masks = train_test_split(
            images,
            masks,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(
            train_images,
            train_masks,
            test_size=self.val_size,
            random_state=self.random_state
        )

        if self.verbose:
            print("training samples", len(self.train_images))
            print("validation samples", len(self.val_images))
            print("test samples", len(self.test_images))

        # datasets

        self.train_ds = SBSegmentationDataset(
            train_images, train_masks, num_classes=self.num_classes)
        self.val_ds = SBSegmentationDataset(
            train_images, train_masks, num_classes=self.num_classes)
        self.trian_ds = SBSegmentationDataset(
            train_images, train_masks, num_classes=self.num_classes)

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
