from pytorch_eo.utils import read_image
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange


class CategoricalImageDataset(Dataset):
    def __init__(self, images, num_classes, chan=0):
        self.images = images
        self.num_classes = num_classes
        self.chan = chan

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        img = read_image(self.images[ix])  # (H, W, C), np
        if img.ndim == 3:
            img = img[..., self.chan]  # H, W
        oh = (np.arange(self.num_classes) == img[..., None]).astype(
            np.float32
        )  # one hot encoding
        return oh  # H, W, C
