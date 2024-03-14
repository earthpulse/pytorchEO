import os
import glob
from pathlib import Path
import pandas as pd
import lightning as L
from pytorch_eo.datasets import RGBImageDataset
from pytorch_eo.datasets import ConcatDataset
from torch.utils.data import DataLoader
from ...utils import download_eotdl, unzip_file
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class DeepGlobeRoadExtraction(L.LightningDataModule):
    # This dataset already comes split into train, validation and test sets
    # However, the dataset comes from a challenge and no labels are available for the valid and test set
    def __init__(
        self,
        batch_size=32,
        path="data",
        train_trans=None,
        val_trans=None,
        test_trans=None,
        num_workers=0,
        pin_memory=False,
        seed=42,
        verbose=False,
        bands=None,
        download=False,
        test_size=0,
        val_size=0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.path = path
        self.test_size = test_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.verbose = verbose
        self.num_classes = 1  # road or not road
        self.bands = bands
        self.download = download
        self.train_trans = (
            train_trans if train_trans is not None else self.setup_trans(train_trans)
        )
        self.val_trans = (
            val_trans if val_trans is not None else self.setup_trans(val_trans)
        )
        self.test_trans = (
            test_trans if test_trans is not None else self.setup_trans(test_trans)
        )

    def setup(self, stage=None):
        path = self.path + "/DeepGlobeRoadExtraction"
        if self.download or not os.path.isdir(path + "/v1"):
            download_eotdl("DeepGlobeRoadExtraction", self.path)
        files = glob.glob(f"{path}/*")
        if len(files) == 1:  # only v1 folder
            unzip_file(
                path + "/v1/deepglobe.zip",
                self.path + "/DeepGlobeRoadExtraction",
                msg="extracting data ...",
            )
        self.df = pd.read_csv(path + "/metadata.csv")
        self.df = self.df.rename(
            columns={"sat_image_path": "image", "mask_path": "mask"}
        )
        self.df = self.df[self.df.split == "train"].drop(columns=["image_id", "split"])
        # add path
        self.df["image"] = path + "/" + self.df["image"]
        self.df["mask"] = path + "/" + self.df["mask"]
        # get splits
        self.make_splits()
        # datasets
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
        if self.test_size > 0:
            train_df, self.test_df = train_test_split(
                self.df,
                test_size=int(len(self.df) * self.test_size),
                random_state=self.seed,
            )
        else:
            train_df, self.test_df = self.df, None
        if self.val_size > 0:
            self.train_df, self.val_df = train_test_split(
                train_df,
                test_size=int(len(self.df) * self.val_size),
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

    def get_dataset(self, df, trans):
        return ConcatDataset(
            {
                "image": RGBImageDataset(df.image.values),
                "mask": RGBImageDataset(df["mask"].values),
            },
            trans,
        )

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

    def setup_trans(self, trans):
        def norm(x, **kwargs):
            return (x / 255.0).astype("float32")

        def to_grey(x, **kwargs):
            return x[:1, ...]

        if trans is None:
            return (
                A.Compose(
                    [
                        # A.Normalize(0, 1, max_pixel_value=255.0),  # no va con la mask
                        A.Lambda(image=norm, mask=norm),
                        ToTensorV2(
                            transpose_mask=True
                        ),  # convert to float tensor and channel first
                        A.Lambda(mask=to_grey),
                    ]
                )
                if trans is None
                else trans
            )
        return trans
