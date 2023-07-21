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
    """
    SEN12Floods dataset
    """
    def __init__(
        self,
        batch_size=25,
        path='./data',
        processed_data_path='./data/sen12floods',
        test_size=0.2,
        val_size=0.2,
        train_trans=None,
        val_trans=None,
        test_trans=None,
        num_workers=0,
        pin_memory=False,
        seed=42,
        verbose=True,
        bands=S2.RGB,
        sensor=Sensors.S2,
    ):
        """
        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 32
        path : str, optional
            Path to the data folder, by default './data'
        processed_data_path : str, optional
            Path to the processed data folder, by default './data/sen12floods'
        test_size : float, optional
            Proportion of the data to use for testing, by default 0.2
        val_size : float, optional
            Proportion of the data to use for validation, by default 0.2
        train_trans : albumentations.Compose, optional
            Transformations to apply to the training data, by default None
        val_trans : albumentations.Compose, optional
            Transformations to apply to the validation data, by default None
        test_trans : albumentations.Compose, optional
            Transformations to apply to the test data, by default None
        num_workers : int, optional
            Number of workers to use for loading the data, by default 0
        pin_memory : bool, optional
            Whether to pin memory or not, by default False
        seed : int, optional
            Seed for reproducibility, by default 42
        verbose : bool, optional
            Whether to print information or not, by default False
        bands : list, optional
            List with the bands to use, by default S2.RGB
        sensor : Sensors, optional
            Sensor to use, by default Sensors.S2
        """
        super().__init__()
        self.bands = bands if bands is not None else S2.RGB
        self.sensor = sensor
        self.batch_size = batch_size
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
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.verbose = verbose
        self.num_classes = 2
        self.classes = ['NO_FLOODING', 'FLOODING']
        self.num_bands = len(self.bands) if isinstance(self.bands, list) else len(self.bands.value) if isinstance(self.bands.value, list) else 1

    def setup(self, stage=None):
        """
        Setup the dataset
        
        Parameters
        ----------
        stage : str, optional
            Stage of the training, by default None
        """
        if isinstance(self.sensor, list):
            raise NotImplementedError(
                "Multiple sensors not implemented for SEN12Floods, please choose one sensor"
            )
        if os.path.exists(self.processed_data_path):
            uncompressed_data_path = self.processed_data_path
        else:
            raise NotImplementedError(
                "You need to download the data first or give a path to the processed data. See the example in /examples/sen12floods.ipynb"
            )
        if self.sensor.value == "S1":
            uncompressed_data_path = os.path.join(uncompressed_data_path, 'sen12floods_s1_source')
        elif self.sensor.value == "S2":
            uncompressed_data_path = os.path.join(uncompressed_data_path, 'sen12floods_s2_source')
        try:
            self.df = pd.read_csv(os.path.join(uncompressed_data_path, 'df.csv'))
        except:
            images_labels = generate_classes_list(uncompressed_data_path)
            self.df = generate_df(images_labels, self.verbose)
            self.df.to_csv(os.path.join(uncompressed_data_path, 'df.csv'), index=False)
        self.make_splits()
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

    def make_splits(self):
        """
        Make the train, validation and test splits
        
        Returns
        -------
        train_df
            Dataframe with the training samples
        val_df
            Dataframe with the validation samples
        test_df
            Dataframe with the test samples
        """
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

    def get_dataset(self, df, trans=None):
        """
        Get the dataset
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the samples
        trans : albumentations.Compose, optional
            Transformations to apply to the data, by default None
        
        Returns
        -------
        ConcatDataset
            Dataset with the samples
        """
        images_ds = self.get_image_dataset(df.image.values)
        return ConcatDataset({"image": images_ds, "label": df.label.values}, trans)

    def get_dataloader(self, ds, batch_size=None, shuffle=False):
        """
        Get the dataloader
        
        Parameters
        ----------
        ds : torch.utils.data.Dataset
            Dataset with the samples
        batch_size : int, optional
            Batch size, by default None
        shuffle : bool, optional
            Whether to shuffle the data or not, by default False
        
        Returns
        -------
        DataLoader
            Dataloader with the samples
        """
        return DataLoader(
            ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )

    def train_dataloader(self, batch_size=None, shuffle=True):
        """
        Get the training dataloader
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default None
        shuffle : bool, optional
            Whether to shuffle the data or not, by default True
        
        Returns
        -------
        DataLoader
            Dataloader with the training samples
        """
        return self.get_dataloader(self.train_ds, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        """
        Get the validation dataloader

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default None
        shuffle : bool, optional
            Whether to shuffle the data or not, by default False
        
        Returns
        -------
        DataLoader
            Dataloader with the validation samples
        """
        return (
            self.get_dataloader(self.val_ds, batch_size, shuffle)
            if self.val_ds is not None
            else None
        )

    def test_dataloader(self, batch_size=None, shuffle=False):
        """
        Get the test dataloader
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default None
        shuffle : bool, optional
            Whether to shuffle the data or not, by default False
            
        Returns
        -------
        DataLoader
            Dataloader with the test samples
        """
        return (
            self.get_dataloader(self.test_ds, batch_size, shuffle)
            if self.test_ds is not None
            else None
        )

    def get_image_dataset(self, images):
        """
        Get the image dataset
        
        Parameters
        ----------
        images : list
            List with the images
        
        Returns
        -------
        SensorImageDataset
            Dataset with the images
        """
        return SingleBandImageDataset(images, self.sensor, self.bands)

    def setup_trans(self, trans=None): 
        """
        Setup the transformations

        Parameters
        ----------
        trans : albumentations.Compose, optional
            Transformations to apply to the data, by default None
        
        Returns
        -------
        albumentations.Compose
            Transformations to apply to the data
        """
        if trans is None:
            def clip(x, **kwargs):
                return np.clip(x, 0.0, 1.0)
    
            def add_channel(x, **kwargs):
                return rearrange(x, "h w -> h w 1") if x.ndim == 2 else x
            
            if self.sensor.value == "S1":
                return (
                    A.Compose(
                        [   
                            A.Resize(512, 512), # scenes have different sizes
                            A.Lambda(image=add_channel),  # add channel dimension if only one band
                            ToTensorV2(),  # convert to float tensor and channel first
                        ]
                    )
                    if trans is None
                    else trans
                )
            else:
                return (
                    A.Compose(
                        [   
                            A.Resize(512, 512),
                            A.Normalize(0, 1, max_pixel_value=4000),  # divide by 4000
                            A.Lambda(image=clip),  # clip to [0,1]
                            A.Lambda(image=add_channel),  # add channel dimension if only one band
                            ToTensorV2(),  # convert to float tensor and channel first
                        ]
                    )
                    if trans is None
                    else trans
                )
        return trans
