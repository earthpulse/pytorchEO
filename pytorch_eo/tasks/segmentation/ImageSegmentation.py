import torch
import torch.nn.functional as F
from einops import rearrange

from pytorch_eo.metrics.segmentation import iou
from ..BaseTask import BaseTask


class ImageSegmentation(BaseTask):
    def __init__(
        self,
        model,
        hparams=None,
        inputs=["image"],
        outputs=["mask"],
        loss_fn=None,
        metrics=None,
    ):

        # defaults
        loss_fn = torch.nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn
        hparams = {"optimizer": "Adam"} if hparams is None else hparams
        metrics = {"iou": iou} if metrics is None else metrics

        super().__init__(model, hparams, inputs, outputs, loss_fn, metrics)

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            x = self.my_prepare_data(batch)
            y_hat = self(x)
            return torch.sigmoid(y_hat)
