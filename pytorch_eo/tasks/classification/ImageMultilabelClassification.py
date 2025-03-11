from pytorch_eo.metrics.classification import accuracy
from ..BaseTask import BaseTask
import torch


class ImageMultilabelClassification(BaseTask):
    def __init__(
        self,
        model,
        hparams=None,
        inputs=["image"],
        outputs=["labels"],
        loss_fn=None,
        metrics=None,
    ):

        # defaults
        loss_fn = torch.nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn
        hparams = {"optimizer": "Adam"} if hparams is None else hparams
        metrics = {"acc": accuracy} if metrics is None else metrics

        super().__init__(model, hparams, inputs, outputs, loss_fn, metrics)

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            x = self.my_prepare_data(batch)
            y_hat = self(x)
            return torch.softmax(y_hat, axis=1)
