import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, data, trans=None, image_key="image"):
        self.data = data
        self.trans = trans
        self.image_key = image_key
        # assert all datasets in data have the same length
        f = len(data[next(iter(data))])
        assert all(
            len(x) == f for x in data.values()
        ), "Datasets must have the same length"
        if trans and image_key not in self.data.keys():
            raise Exception("a valid image field must be provided to albumentations")

    def __len__(self):
        return min(len(v) for v in self.data.values())

    def __getitem__(self, ix):
        data = {k: v[ix] for k, v in self.data.items()}
        transformed = self.apply_transforms(data)
        return transformed if transformed else data

    def apply_transforms(self, data):
        if self.trans:
            if "image" not in self.data.keys():
                data["image"] = data[
                    self.image_key
                ]  # extra transform, can we fix this ?
            trans = self.trans(**data)
            data = {k: trans[k] for k in self.data.keys()}
            return data
        return None
