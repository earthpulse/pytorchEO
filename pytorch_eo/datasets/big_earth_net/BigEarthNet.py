import pytorch_lightning as pl
import pandas as pd
import os
from tqdm import tqdm
from .utils import *
from pytorch_eo.datasets.SingleBandImageDataset import SingleBandImageDataset
from pytorch_eo.datasets.sensors import Sensors, S1
from pytorch_eo.datasets import ConcatDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class BigEarthNet(pl.LightningDataModule):

    # THIS DATASET NEEDS TO BE DOWNLOADED THROUGH http://bigearth.net/

    def __init__(
        self,
        batch_size=32,
        path="data/BigEarthNet",
        s1_folder="BigEarthNet-S1-v1.0",
        s2_folder="BigEarthNet-v1.0",
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
        processed_data_path="data/BigEarthNet",
        filter_clouds=False,
        filter_snow=False,
        clouds_file_name="patches_with_cloud_and_shadow.csv",
        snow_file_name="patches_with_seasonal_snow.csv",
        check_integrity=False, # check labels in metadata match for both S1 and S2 
    ):
        super().__init__()
        self.path = Path(path)
        self.s1_folder = s1_folder
        self.s2_folder = s2_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.verbose = verbose
        self.bands = bands
        self.sensors = list(self.bands.keys())
        self.pin_memory = pin_memory
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.label_groups = label_groups
        self.processed_data_path = Path(processed_data_path)
        self.filter_clouds = filter_clouds
        self.filter_snow = filter_snow
        self.check_integrity = check_integrity
        self.clouds_file_name = Path(clouds_file_name)
        self.snow_file_name = Path(snow_file_name)
        assert isinstance(self.sensors, list), "sensors should be a list"
        self.sensors = set(self.sensors)  # remove duplicates
        assert (
            len(self.sensors) >= 1 and len(self.sensors) <= 2
        ), "BigEarthNet dataset can only work with 1 or 2 sensors (S1, S2 or both)"
        for sensor in self.sensors:
            assert isinstance(sensor, Sensors), "invalida sensor"
            assert sensor in [Sensors.S1, Sensors.S2], "only S1 or S2 sensors are valid"
        if label_groups:
            self.classes = list(self.label_groups.keys())
        else:
            self.classes = LABELS
        self.num_classes = len(self.classes)
        self.in_chans = sum([len(self.bands[sensor]) for sensor in self.sensors])

    def pre_process(self, processed_name):
        s1_path = self.path / self.s1_folder
        s2_path = self.path / self.s2_folder
        data = {}
        # get all s1 images
        if Sensors.S1 in self.sensors:
            data.update({"s1_images": os.listdir(s1_path)})
            if self.verbose: 
                print(f'Number of images in {s1_path}: {len(data["s1_images"])}')
        # get all s2 images
        if Sensors.S2 in self.sensors:
            data.update({"s2_images": os.listdir(s2_path)})
            if self.verbose:
                print(f'Number of images in {s2_path}: {len(data["s2_images"])}')
        # make sure the number of images for S1 and S2 match
        if len(self.sensors) > 1:
            assert len(data["s2_images"]) == len(
                data["s1_images"]
            ), "the number of images for S1 and S2 should match"
        # parse labels from image metadata 
        print("Parsing labels from images metadata ...")    
        patches_folders = data["s1_images"] if Sensors.S1 in self.sensors else data["s2_images"]
        path = s1_path if Sensors.S1 in self.sensors else s2_path
        args = [(path, folder, s2_path, len(self.sensors) > 1, self.check_integrity) for folder in patches_folders]
        # with ThreadPoolExecutor() as pool:
        #     with tqdm(total=len(args)) as progress:
        #         futures = []
        #         for arg in args:
        #             future = pool.submit(parse_labels, arg) # enviamos la tupla de argumentos
        #             future.add_done_callback(lambda p: progress.update())
        #             futures.append(future)
        #         results = []
        #         for future in futures:
        #             result = future.result()
        #             results.append(result)
        results = []
        for arg in tqdm(args):
            results.append(parse_labels(arg))
        labels, s2_patches = zip(*results)
        data.update({"labels": labels})
        if len(self.sensors) > 1:
            data["s2_images"] = s2_patches
        # group labels
        print("Grouping and Encoding labels ...")
        if self.label_groups:
            data["labels"] = group_labels(data["labels"], self.label_groups)
            data["encoded_labels"] = encode_labels(
                data["labels"], list(self.label_groups.keys())
            )
        else:
            data["encoded_labels"] = encode_labels(data["labels"], LABELS)
        self.df = pd.DataFrame({k: v for k, v in data.items()})
        # generate full image paths
        if Sensors.S1 in self.sensors:
            self.df.s1_images = self.df.s1_images.apply(lambda f: str(s1_path / f))
        if Sensors.S2 in self.sensors:
            self.df.s2_images = self.df.s2_images.apply(lambda f: str(s2_path / f))
        # save dataframe
        file_name = get_processed_data_filename(processed_name, self.label_groups)
        print("Saving ...", file_name)
        os.makedirs(self.processed_data_path, exist_ok=True)
        self.df.to_json(self.processed_data_path / file_name)

    def setup(self, stage=None):
        # load pre-procesed data (generate if not found)
        processed_name = "processed"
        for sensor in self.sensors:
            processed_name += f"_{sensor.value}"
        file_name = get_processed_data_filename(processed_name, self.label_groups)
        if not os.path.isfile(self.processed_data_path / file_name):
            print("Pre-processed data not found, generating ...")
            self.pre_process(processed_name)
        print("Loading ...", file_name)
        self.df = pd.read_json(self.processed_data_path / file_name)
        # drop clouds and snow
        if self.filter_clouds and Sensors.S2 in self.sensors:
            clouds = pd.read_csv(self.path / self.clouds_file_name, header=None)
            self.df = self.df[
                ~self.df.s2_images.apply(lambda x: x.split("/")[-1]).isin(
                    clouds[0].values
                )
            ]
        if self.filter_snow:
            snow = pd.read_csv(self.path / self.snow_file_name, header=None)
            self.df = self.df[
                ~self.df.s2_images.apply(lambda x: x.split("/")[-1]).isin(
                    snow[0].values
                )
            ]
        self.make_splits()
        self.train_ds = self.build_dataset(self.train_df, self.train_trans)
        if self.test_size:
            self.test_ds = self.build_dataset(self.test_df, self.test_trans)
        if self.val_size:
            self.val_ds = self.build_dataset(self.val_df, self.val_trans)

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

    def build_dataset(self, df, trans):
        if Sensors.S1 in self.sensors and len(self.sensors) == 1:
            data = {
                "image": SingleBandImageDataset(
                    df.s1_images.values,
                    Sensors.S1,
                    self.bands[Sensors.S1],
                    prefix=[img.split("/")[-1] + "_" for img in df.s1_images.values],
                )
            }
        elif Sensors.S2 in self.sensors and len(self.sensors) == 1:
            data = {
                "image": SingleBandImageDataset(
                    df.s2_images.values,
                    Sensors.S2,
                    self.bands[Sensors.S2],
                    prefix=[img.split("/")[-1] + "_" for img in df.s2_images.values],
                )
            }
        else:
            data = {
                Sensors.S1.value: SingleBandImageDataset(
                    df.s1_images.values,
                    Sensors.S1,
                    self.bands[Sensors.S1],
                    prefix=[img.split("/")[-1] + "_" for img in df.s1_images.values],
                ),
                Sensors.S2.value: SingleBandImageDataset(
                    df.s2_images.values,
                    Sensors.S2,
                    self.bands[Sensors.S2],
                    prefix=[img.split("/")[-1] + "_" for img in df.s2_images.values],
                ),
            }
        data.update({"labels": np.array([np.array(l) for l in df.encoded_labels.values], dtype=np.float32)})
        return ConcatDataset(
            data, trans, image_key="S1" if len(self.sensors) > 1 else "image"
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