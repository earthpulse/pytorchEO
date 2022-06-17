import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 

from pytorch_eo.datasets import ConcatDataset
from .utils import *


class EuroSATBase(pl.LightningDataModule):

    def __init__(self,
                 batch_size=32,
                 download=True,
                 path="./data",
                 test_size=0.2,
                 val_size=0.2,
                 train_trans=None,
                 val_trans=None,
                 test_trans=None,
                 num_workers=0,
                 pin_memory=False,
                 seed=42,
                 verbose=False,
                 label_ratio=1,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.download = download
        self.path = Path(path)
        self.test_size = test_size
        self.val_size = val_size
        self.num_classes = 10
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.label_ratio = label_ratio
        assert label_ratio > 0 and label_ratio <= 1, 'label_ratio should be in range (0, 1]'
        self.num_workers = num_workers
        self.pin_memory = pin_memory 
        self.seed = seed 
        self.verbose = verbose

    def setup(self, stage=None):
        uncompressed_data_path = download_data(
            self.path,
            self.compressed_data_filename,
            self.data_folder,
            self.download,
            self.url,
            self.verbose
        )
        self.classes = generate_classes_list(uncompressed_data_path)
        assert len(self.classes) == self.num_classes
        self.df = generate_df(
            self.classes, uncompressed_data_path, self.verbose)
        self.make_splits()

        # filter by label ratio
        if self.label_ratio < 1:
            train_labels = self.train_df.label.values
            train_images = self.train_df.image.values
            train_images_ratio, train_labels_ratio = [], []
            unique_labels = np.unique(train_labels)
            for label in unique_labels:
                filter = np.array(train_labels) == label
                ixs = filter.nonzero()[0]
                num_samples = filter.sum()
                ratio_ixs = np.random.choice(
                    ixs, int(self.label_ratio*num_samples), replace=False)
                train_images_ratio += (np.array(train_images)
                                       [ratio_ixs]).tolist()
                train_labels_ratio += (np.array(train_labels)
                                       [ratio_ixs]).tolist()
            self.train_df = pd.DataFrame(
                {'image': train_images_ratio, 'label': train_labels_ratio})
            if self.verbose:
                print("training samples after label ratio filtering:",
                      len(self.train_df))

        self.train_ds = self.get_dataset(self.train_df, self.train_trans)
        self.val_ds = self.get_dataset(self.val_df, self.val_trans) if self.val_df is not None else None
        self.test_ds = self.get_dataset(self.test_df, self.test_trans) if self.test_df is not None else None

    def make_splits(self):
        if self.test_size > 0:
            train_df, self.test_df = train_test_split(
                self.df,
                test_size=int(len(self.df)*self.test_size),
            )
        else: 
            train_df, self.test_df = self.df, None
        if self.val_size > 0:
            self.train_df, self.val_df = train_test_split(
                train_df,
                test_size=int(len(self.df)*self.val_size),
            )
        else: 
            self.train_df, self.val_df = train_df, None
        if self.verbose:
            print("Training samples", len(self.train_df))
            if self.val_df is not None:
                print("Validation samples", len(self.val_df))
            if self.test_df is not None:
                print("Test samples", len(self.test_df))
                
    def get_dataset(self, df, trans=None):
        images_ds = self.get_image_dataset(df.image.values) 
        return ConcatDataset({'image': images_ds, 'label': df.label.values}, trans)
            
    def get_image_dataset(self):
        raise NotImplementedError()

    def get_dataloader(self, ds, batch_size=None, shuffle=False):
        return DataLoader(
            ds, 
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle
        )

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.train_ds, batch_size, shuffle)
    
    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.val_ds, batch_size, shuffle) if self.val_ds is not None else None

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.test_ds, batch_size, shuffle) if self.test_ds is not None else None