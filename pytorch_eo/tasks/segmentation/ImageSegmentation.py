import torch

from pytorch_eo.metrics.segmentation import iou
from ..BaseTask import BaseTask


class ImageSegmentation(BaseTask):

    def __init__(self, model, hparams=None, metrics=None):
        # default hparams
        if hparams is None:
            hparams = {
                'loss': 'BCEWithLogitsLoss',
                'optimizer': 'Adam'
            }
        # default metrics
        if metrics is None:
            metrics = {'iou': iou}
        super().__init__(model, hparams, metrics)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x.to(self.device))
            return torch.sigmoid(preds)
