import torch


def accuracy(y_hat, y):
    return (torch.argmax(y_hat, axis=1) == y).sum() / y.shape[0]
