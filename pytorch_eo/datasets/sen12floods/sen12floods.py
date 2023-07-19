from einops import rearrange
import lightning as L
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_eo.datasets.sensors import S2


from pytorch_eo.datasets import ConcatDataset, SensorImageDataset, SingleBandImageDataset
from pytorch_eo.datasets.sensors.sensors import Sensors
from .utils import *


class SEN12Floods(L.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        download=True,
        path='./data',
        processed_data_path='/Users/fran/Documents/datasets/sen12floods/sen12floods/sen12floods_s2_source',
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
        label_ratio=1,
    ):
        super().__init__()
        # self.url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
        # self.compressed_data_filename = "EuroSATallBands.zip"
        # self.data_folder = "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
        self.bands = bands if bands is not None else S2.RGB
        self.sensors = list(self.bands.keys())
        self.num_bands = (
            len(self.bands)
            if isinstance(self.bands, list)
            else None
        )
        self.batch_size = batch_size
        self.download = download
        self.path = Path(path)
        self.processed_data_path = Path(processed_data_path)
        self.test_size = test_size
        self.val_size = val_size
        self.train_trans = (
            train_trans if train_trans is not None else self.setup_trans(train_trans)
        )
        self.val_trans = (
            val_trans if val_trans is not None else self.setup_trans(val_trans)
        )
        self.test_trans = (
            test_trans if test_trans is not None else self.setup_trans(test_trans)
        )
        self.label_ratio = label_ratio
        assert (
            label_ratio > 0 and label_ratio <= 1
        ), "label_ratio should be in range (0, 1]"
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.verbose = verbose

    def setup(self, stage=None):
        if os.path.exists(self.processed_data_path):
            uncompressed_data_path = self.processed_data_path
        else:
            pass
            # TODO: download data
            # uncompressed_data_path = download_data(
            #     self.path,
            #     self.compressed_data_filename,
            #     self.data_folder,
            #     self.download,
            #     self.url,
            #     self.verbose,
            # )
        mosaic_images(uncompressed_data_path, verbose=self.verbose)
        self.classes, images_labels = generate_classes_list(uncompressed_data_path)
        self.num_classes = len(self.classes)
        self.df = generate_df(images_labels, self.verbose)
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
                    ixs, int(self.label_ratio * num_samples), replace=False
                )
                train_images_ratio += (np.array(train_images)[ratio_ixs]).tolist()
                train_labels_ratio += (np.array(train_labels)[ratio_ixs]).tolist()
            self.train_df = pd.DataFrame(
                {"image": train_images_ratio, "label": train_labels_ratio}
            )
            if self.verbose:
                print(
                    "training samples after label ratio filtering:", len(self.train_df)
                )


        self.train_ds = self.get_dataset(self.train_df, self.train_trans)
        self.val_ds = (
            self.get_dataset(self.val_df, self.val_trans)
            if self.val_df is not None
            else None
        )
        self.test_ds = (
            self.get_dataset(self.test_df, self.test_trans)
            if self.test_df is not None
            else None
        )
        
        '''
        self.train_ds = self.build_dataset(self.train_df, self.train_trans)
        if self.test_size:
            self.test_ds = self.build_dataset(self.test_df, self.test_trans)
        if self.val_size:
            self.val_ds = self.build_dataset(self.val_df, self.val_trans)
        '''

    def make_splits(self):
        if self.test_size > 0:
            train_df, self.test_df = train_test_split(
                self.df,
                test_size=int(len(self.df) * self.test_size),
                stratify=self.df.label.values,
                random_state=self.seed,
            )
        else:
            train_df, self.test_df = self.df, None
        if self.val_size > 0:
            self.train_df, self.val_df = train_test_split(
                train_df,
                test_size=int(len(self.df) * self.val_size),
                stratify=train_df.label.values,
                random_state=self.seed,
            )
        else:
            self.train_df, self.val_df = train_df, None
        if self.verbose:
            print("Training samples", len(self.train_df))
            if self.val_df is not None:
                print("Validation samples", len(self.val_df))
            if self.test_df is not None:
                print("Test samples", len(self.test_df))

    def build_dataset(self, df, trans):
        s1_df = df[df['image'].str.contains('_s1_')]
        s2_df = df[df['image'].str.contains('_s2_')]

        # Compare the size of both dataframes. If they are different, sample the biggest one to match the size of the smallest one
        if len(s1_df) != len(s2_df):
            if len(s1_df) > len(s2_df):
                s1_df = s1_df.sample(len(s2_df))
            else:
                s2_df = s2_df.sample(len(s1_df))

        data = {
            Sensors.S1.value: SingleBandImageDataset(
                s1_df,
                Sensors.S1,
                self.bands[Sensors.S1],
                prefix=[img.split("/")[-1] + "_" for img in s1_df['image'].values],
            ),
            Sensors.S2.value: SingleBandImageDataset(
                s2_df,
                Sensors.S2,
                self.bands[Sensors.S2],
                prefix=[img.split("/")[-1] + "_" for img in s2_df['image'].values],
            ),  
        }
        return ConcatDataset(
            data, trans, image_key="S1" if len(self.sensors) > 1 else "image"
        )

    def get_dataset(self, df, trans=None):
        images_ds = self.get_image_dataset(df.image.values)
        return ConcatDataset({"image": images_ds, "label": df.label.values}, trans)

    def get_dataloader(self, ds, batch_size=None, shuffle=False):
        return DataLoader(
            ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.train_ds, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return (
            self.get_dataloader(self.val_ds, batch_size, shuffle)
            if self.val_ds is not None
            else None
        )

    def test_dataloader(self, batch_size=None, shuffle=False):
        return (
            self.get_dataloader(self.test_ds, batch_size, shuffle)
            if self.test_ds is not None
            else None
        )

    def get_image_dataset(self, images):
        return SensorImageDataset(images, Sensors.S2, self.bands)

    def setup_trans(self, trans):
        if trans is None:

            def clip(x, **kwargs):
                return np.clip(x, 0.0, 1.0)

            def add_channel(x, **kwargs):
                return rearrange(x, "h w -> h w 1") if x.ndim == 2 else x

            return (
                A.Compose(
                    [
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.Normalize(0, 1, max_pixel_value=4000),  # divide by 4000
                        A.Lambda(image=clip),  # clip to [0,1]
                        A.Lambda(
                            image=add_channel
                        ),  # add channel dimension if only one band
                        ToTensorV2(),  # convert to float tensor and channel first
                    ]
                )
                if trans is None
                else trans
            )
        return trans
