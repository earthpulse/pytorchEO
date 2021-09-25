import os
import glob
from pathlib import Path
from pytorch_eo.utils.datasets.SBSegmentationDataset import SBSegmentationDataset
import pandas as pd
from ..S2Dataset import S2Dataset
from ...utils.read_image import read_ms_image


class Dataset(SBSegmentationDataset):

    def __init__(self, images, masks, trans, bands, num_classes, norm_value):
        super().__init__(images, masks, trans, bands, num_classes, norm_value)

    def _read_mask(self, mask):
        return read_ms_image(mask, 1).long().squeeze(0)  # H, W


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
                 train_sampler=None,
                 test_sampler=None,
                 val_sampler=None,
                 test_size=0.2,
                 val_size=0.2,
                 num_workers=0,
                 pin_memory=False,
                 seed=42,
                 verbose=False,
                 trans=None,
                 bands=None,
                 ):
        super().__init__(batch_size, train_sampler, test_sampler, val_sampler,
                         test_size, val_size, verbose, num_workers, pin_memory, seed)
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
        self.trans = trans
        self.dataset = dataset
        self.norm_value = norm_value

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

        if self.dataset:
            self.ds = self.dataset(
                images, masks, self.trans, self.bands, self.num_classes, self.norm_value)
        else:
            self.ds = Dataset(images, masks, self.trans,
                              self.bands, self.num_classes, self.norm_value)

        self.make_splits()

    # def uncompress(self, compressed_data_filename, data_folder, msg):
    #     compressed_data_path = self.path / compressed_data_filename
    #     uncompressed_data_path = self.path / data_folder
    #     if not os.path.isdir(uncompressed_data_path):
    #         untar_file(compressed_data_path, self.path, msg=msg)
    #     else:  # TODO: Validate data is correct
    #         pass
