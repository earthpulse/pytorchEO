from pytorch_eo.utils import read_image
from torch.utils.data import Dataset
import torch

# load RGB images


class RGBImageDataset(Dataset):
    def __init__(self, images, norm_value=255):
        self.images = images
        self.norm_value = norm_value

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        return read_image(self.images[ix])  # (H, W, C), np
