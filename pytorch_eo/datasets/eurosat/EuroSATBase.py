from pathlib import Path

from .utils import *
from ..BaseDataset import BaseDataset

from ...utils.datasets.ConcatDataset import ConcatDataset


class EuroSATBase(BaseDataset):

    def __init__(self,
                 batch_size=32,
                 download=True,
                 path="./data",
                 train_sampler=None,
                 test_sampler=None,
                 val_sampler=None,
                 test_size=0.2,
                 val_size=0.2,
                 num_workers=0,
                 pin_memory=False,
                 seed=42,
                 verbose=False,
                 trans=None
                 ):
        super().__init__(batch_size, train_sampler, test_sampler, val_sampler,
                         test_size, val_size, verbose, num_workers, pin_memory, seed)
        self.download = download
        self.path = Path(path)
        self.num_classes = 10
        self.trans = trans

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
        self.images, self.labels = generate_lists(
            self.classes, uncompressed_data_path, self.verbose)
        self.ds = self.build_datasets(self.images, self.labels)
        self.make_splits()

    def build_datasets(self, images, labels):
        images_ds = self.get_image_ds(images)
        labels_ds = labels
        assert len(images_ds) == len(
            labels_ds), 'datasets should have same length'
        return ConcatDataset(
            {'image': images_ds},  # inputs
            {'label': labels_ds},  # outputs
            self.trans  # transforms
        )
