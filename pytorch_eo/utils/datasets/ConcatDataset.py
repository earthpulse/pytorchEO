import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, data, trans=None, image_field='image'):
        self.data = data
        self.trans = trans
        self.image_field = image_field
        if 'image' not in self.data.keys():
            assert self.image_field != 'image', 'a valid image field must be provided to albumentations'
                
    def __len__(self):
        return min(len(v) for v in self.data.values())

    def __getitem__(self, ix):
        data = {k: v[ix] for k, v in self.data.items()}
        if self.trans:
            data['image'] = data[self.image_field] # extra transform, can we fix this ?
            trans = self.trans(**data)
            data = {k: trans[k] for k in self.data.keys()}
            return data
        return data
