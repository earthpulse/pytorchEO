from pathlib import Path
import pandas as pd
import json
import numpy as np
import cv2
from pytorch_eo.datasets import RGBImageDataset, ConcatDataset
from torch.utils.data import DataLoader
import lightning as L
from sklearn.model_selection import train_test_split

class VLDDatasetDetection(L.LightningDataModule):
    def __init__(self, path='data', test_size=0.2, val_size=0.2, batch_size=32, num_workers=4, pin_memory=True, verbose=True):
        super().__init__()
        self.path = Path(path)
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose

    def prepare_data(self):
        pass  

    def setup(self, stage=None):
        with open(self.path / 'train_bbox_annotations.json') as f:
            data = json.load(f)

        images, bboxes, labels = [], [], []
        for item in data["images"]:
            img_path = str(self.path / 'train_bbox_images' / item["file_name"])
            for ann in item["annotations"]:
                images.append(img_path)
                bboxes.append(ann["bbox"])  # formato [x, y, w, h]
                labels.append(0)  # solo hay una clase: "vacant_lot"

        self.df = pd.DataFrame({'image': images, 'bbox': bboxes, 'label': labels})
        self.make_splits()
        self.train_ds = self.get_dataset(self.train_df)
        self.val_ds = self.get_dataset(self.val_df) if self.val_df is not None else None
        self.test_ds = None

    def make_splits(self):
        train_df, test_df = train_test_split(self.df, test_size=self.test_size)
        self.train_df, self.val_df = train_test_split(train_df, test_size=self.val_size)

        if self.verbose:
            print("Train samples:", len(self.train_df))
            print("Val samples:", len(self.val_df))
            print("Test samples:", len(self.test_df) if hasattr(self, "test_df") and self.test_df is not None else "N/A")

    def get_dataset(self, df):
        images_ds = RGBImageDataset(df.image.values)
        return ConcatDataset({
            "image": images_ds,
            "bbox": df.bbox.values,
            "label": df.label.values
        })

    def get_dataloader(self, ds, shuffle=False):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def train_dataloader(self):
        return self.get_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_ds)
    


class VLDSegmentationDataset(L.LightningDataModule):
    def __init__(self, path='data', test_size=0.2, val_size=0.2, batch_size=16, num_workers=4, pin_memory=True, verbose=True):
        super().__init__()
        self.path = Path(path)
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        with open(self.path / 'train_segmentation_annotations.json') as f:
            data = json.load(f)

        images, masks = [], []

        for item in data["images"]:
            fname = item["file_name"]
            img_path = str(self.path / 'train_segmentation_images' / fname)
            width, height = item["width"], item["height"]

            # Crear máscara binaria
            mask = np.zeros((height, width), dtype=np.uint8)
            for ann in item["annotations"]:
                coords = np.array(ann["segmentation"]).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [coords], color=1)

            images.append(img_path)
            masks.append(mask)

        self.df = pd.DataFrame({'image': images, 'mask': masks})
        self.make_splits()
        self.train_ds = self.get_dataset(self.train_df)
        self.val_ds = self.get_dataset(self.val_df)

    def make_splits(self):
        train_df, val_df = train_test_split(self.df, test_size=self.val_size)
        self.train_df, self.val_df = train_df, val_df
        if self.verbose:
            print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    def get_dataset(self, df):
        images_ds = RGBImageDataset(df.image.values)
        return ConcatDataset({
            "image": images_ds,
            "mask": df["mask"].values
        })

    def get_dataloader(self, ds, shuffle=False):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def train_dataloader(self):
        return self.get_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_ds)