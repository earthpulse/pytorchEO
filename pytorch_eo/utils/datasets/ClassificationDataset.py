from pytorch_eo.utils import read_image
from torch.utils.data import Dataset
import torch


class ClassificationDataset(Dataset):
    def __init__(self, images, labels=None, trans=None, norm_value=255.):
        self.images = images
        self.labels = labels
        self.trans = trans
        self.norm_value = norm_value

    def __len__(self):
        return len(self.images)

    def _read_image(self, img):
        pass

    def _norm_image(self, img):
        return (img / self.norm_value)

    def _trans_image(self, img):
        if self.trans:  # albumentations
            img_trans = self.trans(image=img)['image']
            return img_trans
        return img

    def _to_tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)

    def __getitem__(self, ix):
        img = self._read_image(self.images[ix])
        img_norm = self._norm_image(img)
        img_trans = self._trans_image(img_norm)  # on normalized img
        img_t = self._to_tensor(img_trans)
        if self.labels:
            label_t = torch.tensor(self.labels[ix], dtype=torch.long)
            return img_t, label_t
        return img_t
