import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, data, trans=None):
        self.data = data
        self.trans = trans

    def __len__(self):
        return min(len(v) for v in self.data.values())

    def __getitem__(self, ix):
        data = {k: v[ix] for k, v in self.data.items()}
        if self.trans:
            trans = self.trans(**data)
            data = {k: trans[k] for k in self.data.keys()}
            return data
        return data
