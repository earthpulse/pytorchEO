import os
import glob
from pathlib import Path
import pandas as pd

from ...utils.datasets.SingleBandImageDataset import SingleBandImageDataset
from ...utils.sensors import Sensors
from ...utils.datasets.ConcatDataset import ConcatDataset
from ...utils.datasets.CategoricalImageDataset import CategoricalImageDataset
from ..BaseDataset import BaseDataset


class LandCoverNet(BaseDataset):

    # THIS DATASET NEEDS TO BE DOWNLOADED THROUGH https://registry.mlhub.earth/10.34911/rdnt.d2ce8i/
    # CAN WE HAVE A PUBLIC LINK ?

    # WE HAVE 1980 LABELS, DERIVED FROM S2 TIME SERIES
    # THIS DATASET ASSIGNS THE SAME MASK TO ALL IMAGES IN THE SAME TIME SERIES
    # in the future we will make datasets with time series

    def __init__(self,
                 batch_size,
                 path='/data',
                 compressed_data_filename='ref_landcovernet_v1_source.tar',
                 compressed_labels_filename='ref_landcovernet_v1_labels.tar',
                 data_folder='ref_landcovernet_v1_source',
                 labels_folder='ref_landcovernet_v1_labels',
                 test_size=0.2,
                 val_size=0.2,
                 train_trans=None,
                 val_trans=None,
                 test_trans=None,
                 num_workers=0,
                 pin_memory=False,
                 seed=42,
                 verbose=False,
                 bands=None,
                 ):
        super().__init__(batch_size, test_size, val_size,
                         verbose, num_workers, pin_memory, seed)
        self.path = Path(path)
        self.compressed_data_filename = compressed_data_filename
        self.compressed_labels_filename = compressed_labels_filename
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
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.bands = bands

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

        self.df = pd.DataFrame({'image': images, 'mask': masks})
        self.make_splits()

        self.train_ds = self.build_dataset(self.train_df, self.train_trans)
        if self.test_size:
            self.test_ds = self.build_dataset(self.test_df, self.test_trans)
        if self.val_size:
            self.val_ds = self.build_dataset(self.val_df, self.val_trans)

    def build_dataset(self, df, trans):
        return ConcatDataset({
            'image': SingleBandImageDataset(df.image.values, Sensors.S2, self.bands),
            'mask': CategoricalImageDataset(df['mask'].values, self.num_classes, 0)
        }, trans)
