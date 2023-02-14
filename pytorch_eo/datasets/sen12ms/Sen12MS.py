from pytorch_eo.utils.read_image import read_ms_image
import pytorch_lightning as pl
import os
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_eo.utils.datasets.SegmentationDataset import SegmentationDataset
import numpy as np
from enum import Enum
import pandas as pd
import torch
import torch.nn.functional as F


class Dataset(SegmentationDataset):
    def __init__(self, images, masks=None, classes=[], trans=None, norm_value=4000):
        super().__init__(images, masks, trans, len(classes), norm_value)
        self.classes = classes

    def _norm_image(self, img):
        return np.clip(10.0 ** (img / 10.0), 0, 1)  # undo db, 0-1

    def _img_to_tensor(self, img):
        img_t = torch.from_numpy(img).float().permute(2, 0, 1)

        if torch.any(img_t.isnan()):  # there are nans in some images !
            img_t[img_t != img_t] = 0
            if torch.any(img_t.isnan()):
                raise ValueError("image has nans !")

        return img_t

    def _mask_to_tensor(self, mask):

        # MODIS Land Cover: 4 channels corresponding to IGBP, LCCS Land Cover, LCCS Land Use, and LCCS Surface Hydrology layers.
        # The overall accuracies of the layers are about 67% (IGBP), 74% (LCCS land cover), 81% (LCCS land use), and 87% (LCCS sur- face hydrology), respectively

        mask = mask[..., 2]  # LCCS LU

        # swap class value for position in classes list
        _mask = mask.copy()
        for ix, label in enumerate(self.classes):
            for value in label["values"]:
                _mask[mask == value] = ix

        # one hot encoding
        # mask_oh = F.one_hot(torch.from_numpy(_mask).long(), num_classes=self.num_classes)

        mask_oh = np.arange(self.num_classes) == _mask[..., None]

        mask_oh_t = torch.from_numpy(mask_oh).float().permute(2, 0, 1)

        if torch.any(mask_oh_t.isnan()):
            raise ValueError("mask has nans !")

        return mask_oh_t


class S1Dataset(Dataset):
    def __init__(self, images, masks=None, classes=[], trans=None):
        super().__init__(images, masks, classes, trans)

    def _norm_image(self, img):
        return np.clip(10.0 ** (img / 10.0), 0, 1)  # undo db, 0-1


class S2Dataset(Dataset):
    def __init__(
        self,
        images,
        masks=None,
        classes=[],
        trans=None,
        bands=(3, 2, 1),
        norm_value=4000,
    ):
        super().__init__(images, masks, classes, trans, norm_value)
        self.bands = bands

    def _read_image(self, img):
        return read_ms_image(img, self.bands)

    def _norm_image(self, img):
        return (img / self.norm_value).clip(0, 1)


class LCBands(Enum):
    IGBP = igbp = 1
    LCCS1 = landcover = 2
    LCCS2 = landuse = 3
    LCCS3 = hydrology = 4
    ALL = [IGBP, LCCS1, LCCS2, LCCS3]
    NONE = []


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    S1 = "s1"
    S2 = "s2"
    DF = [S1, S2]


class Sen12MS(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        path="/data",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        verbose=True,
        sensor=Sensor.S2,
        train_trans=None,
        val_trans=None,
        test_trans=None,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.path = Path(path)
        self.num_classes = 10
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.verbose = verbose
        self.sensor = sensor
        self.classes = [  # LCCS LU, extracted from paper
            {"name": "other", "values": [255], "color": "#000000"},
            {"name": "dense-forest", "values": [10], "color": "#277732"},
            {"name": "open-forest", "values": [20, 25], "color": "#56bf64"},
            {"name": "herbaceous", "values": [30, 36, 35], "color": "#00ff00"},
            {"name": "shrublands", "values": [40], "color": "#ffeb14"},
            {"name": "urban", "values": [9], "color": "#ff0000"},
            {"name": "permanent-snow-and-ice", "values": [2], "color": "#ffffff"},
            {"name": "barren", "values": [1], "color": "#888888"},
            {"name": "water-bodies", "values": [3], "color": "#0000ff"},
        ]
        self.in_chans = 2 if self.sensor == Sensor.S1 else 3
        self.num_classes = len(self.classes)
        if self.sensor == Sensor.S1:
            self.Dataset = S1Dataset
        elif self.sensor == Sensor.S2:
            self.Dataset = S2Dataset
        else:
            raise ValueError("Invalid sensor")
        self.train_trans = train_trans
        self.test_trans = test_trans
        self.val_trans = val_trans

    def get_scene_ids(self, season):
        season = Seasons(season).value
        path = os.path.join(self.path, season)

        if not os.path.exists(path):
            raise NameError(
                "Could not find season {} in base directory {}".format(
                    season, self.path
                )
            )

        scene_list = [os.path.basename(s) for s in glob.glob(os.path.join(path, "*"))]
        scene_list = [int(s.split("_")[1]) for s in scene_list]
        return set(scene_list)

    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.path, season, f"s1_{scene_id}")

        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season)
            )

        patch_ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(path, "*"))
        ]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]

        return patch_ids

    def get_season_ids(self, season):
        season = Seasons(season).value
        ids = {}
        scene_ids = self.get_scene_ids(season)

        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)

        return ids

    def setup(self, stage=None):

        season_ids = {}
        for season in Seasons.ALL.value:
            season_ids[season] = self.get_season_ids(season)

        self.season_ids = season_ids

        data = {"image": [], "mask": [], "season": []}
        for season in Seasons.ALL.value:
            scenes = season_ids[season]
            for scene in scenes:
                patches = season_ids[season][scene]
                for patch in patches:
                    data["image"].append(
                        f"{self.path}/{season}/{self.sensor.value}_{scene}/{season}_{self.sensor.value}_{scene}_p{patch}.tif"
                    )
                    data["mask"].append(
                        f"{self.path}/{season}/lc_{scene}/{season}_lc_{scene}_p{patch}.tif"
                    )
                    data["season"].append(season)
        df = pd.DataFrame(data)

        assert len(df) == 180662, "the dataset should contain 180662 patch triplets"

        # data splits (can we stratify ?)
        test_size = int(len(df) * self.test_size)
        val_size = int(len(df) * self.val_size)

        train_df, self.test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df.season,
            shuffle=True,
        )

        self.train_df, self.val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=self.random_state,
            stratify=train_df.season,
            shuffle=True,
        )

        if self.verbose:
            print("training samples", len(self.train_df))
            print("validation samples", len(self.val_df))
            print("test samples", len(self.test_df))

        # datasets

        self.train_ds = self.Dataset(
            self.train_df.image.values,
            self.train_df["mask"].values,
            self.classes,
            self.train_trans,
        )
        self.val_ds = self.Dataset(
            self.val_df.image.values,
            self.val_df["mask"].values,
            self.classes,
            self.val_trans,
        )
        self.test_ds = self.Dataset(
            self.test_df.image.values,
            self.test_df["mask"].values,
            self.classes,
            self.test_trans,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
