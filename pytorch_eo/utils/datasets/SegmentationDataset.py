from pytorch_eo.utils import read_image
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F


class SegmentationDataset(Dataset):
    def __init__(self, images, masks=None, trans=None, num_classes=None, norm_value=255.):
        self.images = images
        self.masks = masks
        self.trans = trans
        self.norm_value = norm_value
        self.num_classes = num_classes

        if self.masks is not None and self.num_classes is None:
            raise ValueError("you have to specify the number of classes !")

    def __len__(self):
        return len(self.images)

    def _read_image(self, img):
        return read_image(img)

    def _read_mask(self, mask):
        return read_image(mask)

    def _norm_image(self, img):
        return (img / self.norm_value)

    def _trans_image_mask(self, img, mask=None):
        if self.trans:  # albumentations by default
            if mask is not None:
                trans = self.trans(image=img, mask=mask)
                return trans['image'], trans['mask']
            return self.trans(image=img)['image']
        return img, mask

    def _mask_to_tensor(self, mask):
        # mask must be long to one hot, but for training we need float
        # C, H, W
        return F.one_hot(mask, num_classes=self.num_classes).permute(2, 0, 1).float()

    def __getitem__(self, ix):
        img = self._read_image(self.images[ix])
        img_norm = self._norm_image(img)
        if self.masks is not None:
            mask = self._read_mask(self.masks[ix])
            img_t, mask_trans = self._trans_image_mask(
                img_norm, mask)  # on normalized img
            mask_t = self._mask_to_tensor(mask_trans)
            return img_t, mask_t
        img_trans = self._trans_image(img_norm)  # on normalized img
        img_t = self._img_to_tensor(img_trans)
        return img_t
