import lightning as L
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
import json
from pytorch_eo.datasets import SensorImageDataset
from pytorch_eo.datasets.sensors import Sensors

try:
    import eotdl
except ImportError:
    raise ImportError("Please install eotdl with `pip install eotdl`")

from eotdl.datasets import download_dataset

from pytorch_eo.datasets import ConcatDataset


class EOTDLDataset(L.LightningDataModule):
    # This dataset only works for Q1+ datasets (https://www.eotdl.com/docs/getting-started/quality)
    # only supports classification (images + one label per image)
    # get in touch with us if you need support for other tasks
    def __init__(
        self,
        dataset_name,  # the name of the dataset in EOTDL
        batch_size=32,
        download=True,
        force=False,
        assets=True,
        version=1,
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
        bands=None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.download = download
        self.path = Path(path)
        self.test_size = test_size
        self.val_size = val_size
        self.num_classes = 10
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
        self.force = force
        self.assets = assets
        self.verbose = verbose
        self.version = version
        self.bands = bands

    def setup(self, stage=None):
        # download dataset
        try:
            download_path = download_dataset(
                self.dataset_name,
                version=self.version,
                path=str(self.path),
                assets=self.assets,
                force=self.force,
                verbose=self.verbose,
            )
            if self.verbose:
                print("Dataset downloaded to", download_path)
        except Exception as e:
            print(str(e))
            download_path = str(
                self.path / self.dataset_name / f"v{self.version}"
            )  # can we get different versions?

        # get list of classes
        try:
            labels_collection = (
                f"{download_path}/{self.dataset_name}/labels/collection.json"
            )
            with open(labels_collection) as f:
                labels = json.load(f)
            self.classes = sorted(
                labels["summaries"]["label:classes"][0]["classes"]
            )  # will this always be the same?
        except Exception as e:
            # maybe there are no labels (unsupervised learning)
            print(str(e))
            self.classes = None

        # get all the assets
        assets = os.listdir(download_path + "/assets")
        images = [f for f in assets if f.endswith(".tif")]
        images = [download_path + "/assets/" + f for f in images]
        if self.classes:
            label_files = [f for f in assets if f.endswith(".geojson")]
            labels, classes = [], []
            for file in label_files:
                with open(download_path + "/assets/" + file) as f:
                    data = json.load(f)
                cls = data["features"][0]["properties"]["label"]
                classes.append(cls)
                labels.append(self.classes.index(cls))  # this will depend on the task
            self.df = pd.DataFrame({"image": images, "label": labels, "class": classes})
        else:
            self.df = pd.DataFrame({"image": images})

        self.train_ds = self.get_dataset(self.df, self.train_trans)

        # TODO: if metadata has splits, use them. Otherwise, we do not split by default.

    def get_dataset(self, df, trans=None):
        # TODO: metadata should tell which sensor to use !!!
        images_ds = SensorImageDataset(df.image.values, Sensors.S2, self.bands)
        if self.classes:  # this will depend on the task
            return ConcatDataset({"image": images_ds, "label": df.label.values}, trans)
        return ConcatDataset({"image": images_ds}, trans)

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

    def setup_trans(self, trans):
        if trans is None:

            # TODO: metadata should hint minimum set of transforms to use (normalize, resize, etc.)

            # def clip(x, **kwargs):
            #     return np.clip(x, 0.0, 1.0)

            # def add_channel(x, **kwargs):
            #     return rearrange(x, "h w -> h w 1") if x.ndim == 2 else x

            return (
                A.Compose(
                    [
                        # A.Lambda(image=clip),  # clip to [0,1]
                        # A.Lambda(image=add_channel),),
                        ToTensorV2(),  # convert to float tensor and channel first
                    ]
                )
                if trans is None
                else trans
            )
        return trans
