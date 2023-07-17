from .EuroSATBase import EuroSATBase
from pytorch_eo.datasets import SensorImageDataset
from pytorch_eo.datasets.sensors import Sensors
import albumentations as A
from albumentations.pytorch import ToTensorV2
from einops import rearrange
import numpy as np
from pytorch_eo.datasets.sensors import S2


class EuroSAT(EuroSATBase):
    def __init__(
        self,
        batch_size=32,
        download=True,
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
        bands=None,
        label_ratio=1,
    ):
        super().__init__(
            batch_size,
            download,
            path,
            test_size,
            val_size,
            train_trans,
            val_trans,
            test_trans,
            num_workers,
            pin_memory,
            seed,
            verbose,
            label_ratio,
        )
        self.url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
        self.compressed_data_filename = "EuroSATallBands.zip"
        self.data_folder = "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
        self.bands = bands if bands is not None else S2.ALL
        self.num_bands = (
            len(self.bands)
            if isinstance(self.bands, list)
            else len(self.bands.value)
            if isinstance(self.bands.value, list)
            else 1
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
