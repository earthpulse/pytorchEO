from pytorch_eo.utils import read_ms_image
from torch.utils.data import Dataset

# load TIF images with rasterio


class RasterioImageDataset(Dataset):
    def __init__(self, images, bands):
        self.images = images
        self.bands = bands

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        return read_ms_image(self.images[ix], self.bands)  # C, H, W
