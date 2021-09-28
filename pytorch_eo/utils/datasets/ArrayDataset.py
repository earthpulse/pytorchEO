import torch


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, a, dtype=None):
        self.a = a
        self.dtype = dtype

    def __len__(self):
        return len(self.a)

    def __getitem__(self, ix):
        sample = self.a[ix]
        if self.dtype:
            return torch.tensor(sample, dtype=self.dtype)
        return torch.tesnor(sample)
