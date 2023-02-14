import torch
from pytorch_eo.utils import read_image


class RGBImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        return read_image(self.images[ix])
