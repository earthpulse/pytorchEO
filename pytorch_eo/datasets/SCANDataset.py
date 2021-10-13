from pathlib import Path
import numpy as np

from .BaseDataset import BaseDataset

from ..utils.datasets.SensorImageDataset import SensorImageDataset
from ..utils.sensors import Sensors
from ..utils.datasets.ConcatDataset import ConcatDataset
from ..utils.datasets.CategoricalImageDataset import CategoricalImageDataset

import requests
import shutil
import os
from tqdm import tqdm
import pandas as pd
import json
import rasterio
import geopandas as gpd
import rasterio.features
from ..config import SCAN_URL, SPAI_URL


class SCANDataset(BaseDataset):

    def __init__(self,
                 dataset,
                 batch_size=32,
                 download=False,
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
                 bands=None
                 ):
        super().__init__(batch_size, test_size, val_size,
                         verbose, num_workers, pin_memory, seed)
        self.dataset = dataset
        self.download = download
        self.path = Path(path) / self.dataset
        self.train_trans = train_trans
        self.val_trans = val_trans
        self.test_trans = test_trans
        self.bands = bands

    def setup(self, stage=None):
        super().setup(stage)

        # call SCAN api to get the basic dataset info
        info_path = self.path / 'info.json'
        try:
            with open(info_path) as json_file:
                self.info = json.load(json_file)
            if self.verbose:
                print("found info.json")
        except:
            if self.verbose:
                print("query scan api")
            self.info = requests.get(
                f'{SCAN_URL}/dataset/{self.dataset}').json()
            with open(info_path, 'w') as outfile:
                json.dump(self.info, outfile)
        self.num_classes = len(self.info['labels'])
        self.classes = self.info['labels']

        # download images and annotations from SPAI
        # TODO: can we do this better ? (stream one zip file?)
        if self.download:
            # get a valid token
            token = requests.get(f'{SCAN_URL}/token').json()['token']
            os.makedirs(self.path, exist_ok=True)
            if self.verbose:
                print("downloading data")
            for image in tqdm(self.info['images']):
                # image
                url = f"{SPAI_URL}/images/{image['id']}?datastore={self.info['id']}"
                response = requests.get(
                    url, headers={'Authorization': 'Bearer ' + token}, stream=True)
                with open(f'{self.path}/{image["id"]}.tif', 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                del response
                # annotations
                url = f"{SPAI_URL}/images/{image['id']}/meta?datastore={self.info['id']}&name=annotations"
                response = requests.get(
                    url, headers={'Authorization': 'Bearer ' + token}, stream=True)
                with open(f'{self.path}/{image["id"]}.geojson', 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                del response

            # convert polys to mask (eurosat test)
            for image in tqdm(self.info['images']):
                ds = rasterio.open(f'{self.path}/{image["id"]}.tif')
                annotations = gpd.read_file(
                    f'{self.path}/{image["id"]}.geojson')
                mask = np.zeros(
                    (*ds.shape, len(self.info['labels'])))
                mask[..., -1] = 1
                for band, label in enumerate(self.info['labels']):
                    polys = annotations[annotations['label'] == label].geometry
                    if len(polys) > 0:
                        mask[..., band] = rasterio.features.rasterize(
                            polys, out_shape=mask.shape[:-1], transform=ds.transform)
                mask = np.argmax(mask, -1)
                with rasterio.Env():
                    profile = ds.profile
                    profile.update(
                        dtype=rasterio.uint8,
                        count=1,
                        compress='lzw'
                    )
                    with rasterio.open(f'{self.path}/{image["id"]}_mask.tif', 'w', **profile) as dst:
                        dst.write(mask.astype(rasterio.uint8), 1)

        self.df = pd.DataFrame({
            'image': [f'{self.path}/{image["id"]}.tif' for image in self.info['images']],
            'mask': [f'{self.path}/{image["id"]}_mask.tif' for image in self.info['images']],
        })

        self.make_splits()
        self.build_datasets()

    def build_dataset(self, df, trans):
        # scan me tiene que decir que poner aquí
        # - tasks
        # - sensors
        return ConcatDataset({
            'image': SensorImageDataset(df.image.values, Sensors.S2, self.bands),
            'mask': CategoricalImageDataset(df['mask'].values, self.num_classes)
        }, trans)
