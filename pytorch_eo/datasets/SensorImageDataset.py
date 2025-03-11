from .RasterioImageDataset import RasterioImageDataset
from pytorch_eo.datasets.sensors import sensors, Sensors, bands2values
import numpy as np

# load tif images with rasterio


class SensorImageDataset(RasterioImageDataset):
    def __init__(self, images, sensor, bands):
        super().__init__(images, bands)

        assert isinstance(sensor, Sensors), "invalid sensor"
        sensor = getattr(sensors, sensor.value)

        # parse bands and compute number
        self.bands = bands
        if bands is None:
            self.bands = sensor.ALL

        if isinstance(self.bands, list):
            self.in_chans = len(bands)
            for band in self.bands:
                assert band in sensor, "invalid band"
        else:
            assert self.bands in sensor, "invalid band"
            if isinstance(self.bands.value, list):
                self.in_chans = len(self.bands.value)
            else:
                self.in_chans = 1

        # convert to values
        self.bands = bands2values(self.bands)

    def __getitem__(self, ix):
        img_data = super().__getitem__(ix)
        # uin16 is not supported by pytorch
        return img_data.astype(np.float32)
