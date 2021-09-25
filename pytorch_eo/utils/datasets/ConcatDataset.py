import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, trans=None):
        self.inputs = inputs
        self.outputs = outputs
        # all datasets should have the same length !
        # no keys should be repeated
        self.trans = trans

    def __len__(self):
        return min(len(d) for d in self.inputs.values())

    def __getitem__(self, ix):
        # aqui tengo arrays np o listas, apply transforms
        # default collate pasa a tensor
        if self.trans:
            inputs = {k: d[ix] for k, d in self.inputs.items()}
            outputs = {k: d[ix] for k, d in self.outputs.items()}
            inputs.update(outputs)  # join dicts
            trans = self.trans(**inputs)
            # parece que puedes sacar el valor sin modificar del trans
            # si no le has dicho que haga trans :)
            inputs = tuple(trans[k] for k in self.inputs.keys())
            outputs = tuple(trans[k] for k in self.outputs.keys())
            return inputs + outputs
        inputs = tuple(d[ix] for d in self.inputs.values())
        outputs = tuple(d[ix] for d in self.outputs.values())
        return inputs + outputs
