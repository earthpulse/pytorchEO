import torch


def iou(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    intersection = torch.sum(gt * pr, axis=(-2, -1))
    union = (
        torch.sum(gt, axis=(-2, -1)) + torch.sum(pr, axis=(-2, -1)) - intersection + eps
    )
    ious = (intersection + eps) / union
    return torch.mean(ious)


def f1_score(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    tp = torch.sum(gt * pr, axis=(-2, -1))
    fp = torch.sum(pr, axis=(-2, -1)) - tp
    fn = torch.sum(gt, axis=(-2, -1)) - tp
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return torch.mean(f1)
