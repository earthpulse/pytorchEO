from pathlib import Path
import numpy as np 

from .utils import *
from ..BaseDataset import BaseDataset

from ...utils.datasets.ConcatDataset import ConcatDataset


class EuroSATBase(BaseDataset):

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
        super().__init__(batch_size, test_size, val_size,
                         verbose, num_workers, pin_memory, seed)
        self.download = download
        self.path = Path(path)
        self.num_classes = 10
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.label_ratio = label_ratio
        assert label_ratio > 0 and label_ratio <= 1, 'label_ratio should be in range (0, 1]'

    def setup(self, stage=None):
        super().setup(stage)
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
        
        self.make_splits(stratify="label")

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
            self.train_df = pd.DataFrame({'image': train_images_ratio, 'label': train_labels_ratio})
            if self.verbose:
                print("training samples after label ratio filtering:", len(self.train_df))

        self.build_datasets()

    def build_dataset(self, df, trans):
        images_ds = self.get_image_ds(df.image.values)
        labels_ds = df.label.values
        assert len(images_ds) == len(
            labels_ds), 'datasets should have same length'
        return ConcatDataset({'image': images_ds, 'label': labels_ds}, trans)

    def build_datasets(self):
        self.train_ds = self.build_dataset(self.train_df, self.train_trans)
        if self.test_size:
            self.test_ds = self.build_dataset(self.test_df, self.test_trans)
        if self.val_size:
            self.val_ds = self.build_dataset(self.val_df, self.val_trans)
