from ..BaseDataset import BaseDataset
import pandas as pd
import os
from tqdm import tqdm 
import json 
from .utils import *
from ...utils.datasets.SingleBandImageDataset import SingleBandImageDataset
from ...utils.sensors import Sensors, S1
from ...utils.datasets.ConcatDataset import ConcatDataset
from pathlib import Path

class BigEarthNet(BaseDataset):

    # THIS DATASET NEEDS TO BE DOWNLOADED THROUGH http://bigearth.net/


    def __init__(self,
                batch_size,
                path='data/BigEarthNet/BigEarthNet-S1-v1.0',
                s1_folder='BigEarthNet-S1-v1.0',
                s2_folder='BigEarthNet-v1.0',
                test_size=0.2,
                val_size=0.2,
                train_trans=None,
                val_trans=None,
                test_trans=None,
                num_workers=0,
                pin_memory=False,
                seed=42,
                verbose=False,
                bands={Sensors.S1: [S1.VV, S1.VH]},
                label_groups=None,
                processed_data_path='data/BigEarthNet'
                ):
        super().__init__(batch_size, test_size, val_size,
                            verbose, num_workers, pin_memory, seed)
        self.path = Path(path)
        self.s1_folder = s1_folder
        self.s2_folder = s2_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.bands = bands
        self.sensors = list(self.bands.keys())
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.label_groups = label_groups
        self.processed_data_path = Path(processed_data_path)

        assert isinstance(self.sensors, list), 'sensors should be a list'
        self.sensors = set(self.sensors) # remove duplicates
        assert len(self.sensors) >= 1 and len(self.sensors) <= 2, 'BigEarthNet dataset can only work with 1 or 2 sensors (S1, S2 or both)'
        for sensor in self.sensors:
            assert isinstance(sensor, Sensors), 'invalida sensor'
            assert sensor in [Sensors.S1, Sensors.S2], 'only S1 or S2 sensors are valid'

        if label_groups:
            self.classes = list(self.label_groups.keys())
        else:
            self.classes = LABELS
        self.num_classes = len(self.classes)

    def pre_process(self, processed_name):

        data = {}

        # get all s1 images
        if Sensors.S1 in self.sensors:
            s1_path = self.path / self.s1_folder
            data.update({'s1_images': os.listdir(s1_path)})
            print(f'Number of images in {s1_path}: {len(data["s1_images"])}')

        # get all s2 images
        if Sensors.S2 in self.sensors:
            s2_path = self.path / self.s2_folder
            data.update({'s2_images': os.listdir(s2_path)})
            print(f'Number of images in {s2_path}: {len(data["s2_images"])}')

        if len(self.sensors) > 1:
            assert len(data["s2_images"]) == len(data["s1_images"]), 'the number of images for S1 and S2 should match'

        # parse labels from image metadata
        print("Parsing labels from images metadata ...")
        if Sensors.S1 in self.sensors:
            patches_folders = data['s1_images']
            path = s1_path
        else:
            patches_folders = data['s2_images']
            path = s2_path
        labels, s2_patches = [], []
        count = 0
        for folder in tqdm(patches_folders):
            with open(path / folder / f'{folder}_labels_metadata.json') as f:
                metadata = json.load(f)
                labels.append(metadata['labels'])
                if len(self.sensors) > 1:
                    s2_image = metadata['corresponding_s2_patch']
                    with open(s2_path / s2_image / f'{s2_image}_labels_metadata.json') as f:
                        s2_metadata = json.load(f)
                        assert len(s2_metadata['labels']) == len(metadata['labels'])
                        assert all(label in s2_metadata['labels']
                                for label in metadata['labels'])
                    s2_patches.append(s2_image)
            count += 1
            if count >= 100:
                break
        data.update({'labels': labels})
        if len(self.sensors) > 1:
            data['s2_images'] = s2_patches

        # group labels
        print("Grouping and Encoding labels ...")
        if self.label_groups:
            data['labels'] = group_labels(
                data['labels'], self.label_groups)
            data['encoded_labels'] = encode_labels(
                data['labels'], list(self.label_groups.keys()))
        else:
            data['encoded_labels'] = encode_labels(
                data['labels'], LABELS)

        print(len(data['s1_images']))

        self.df = pd.DataFrame({
            k: v[:100] for k, v in data.items()
        })

        # generate full image paths
        if Sensors.S1 in self.sensors:
            self.df.s1_images = self.df.s1_images.apply(
                lambda f: str(s1_path / f )
            )
        if Sensors.S2 in self.sensors:
            self.df.s2_images = self.df.s2_images.apply(
                lambda f: str(s2_path / f )
            )

        # save dataframe
        file_name = get_processed_data_filename(
            processed_name, self.label_groups)
        print('Saving ...', file_name)
        self.df.to_json(
            self.processed_data_path / file_name)


    def setup(self, stage=None):

        # load pre-procesed data (generate if not found)
        processed_name = 'processed' 
        for sensor in self.sensors:
            processed_name += f'_{sensor.value}'
        file_name = get_processed_data_filename(
            processed_name, self.label_groups)
        if not os.path.isfile(self.processed_data_path / file_name):
            print("Pre-processed data not found, generating ...")
            self.pre_process(processed_name)
        print('Loading ...', file_name)
        self.df = pd.read_json(
            self.processed_data_path / file_name)

        self.make_splits()

        self.train_ds = self.build_dataset(self.train_df, self.train_trans)
        if self.test_size:
            self.test_ds = self.build_dataset(self.test_df, self.test_trans)
        if self.val_size:
            self.val_ds = self.build_dataset(self.val_df, self.val_trans)

    def build_dataset(self, df, trans):
        images_ds = {}
        if Sensors.S1 in self.sensors:
            images_ds .update({'image': SingleBandImageDataset(df.s1_images.values, Sensors.S1, self.bands[Sensors.S1], prefix=[img.split('/')[-1] + '_' for img in df.s1_images.values])})
        if Sensors.S2 in self.sensors:
            images_ds .update({'image': SingleBandImageDataset(df.s2_images.values, Sensors.S2, self.bands[Sensors.S2], prefix=[img.split('/')[-1] + '_' for img in df.s2_images.values])})
        labels_ds = df.encoded_labels.values
        return ConcatDataset(
            images_ds,  # inputs
            {'labels': labels_ds},  # outputs
            trans  # transforms
        )
