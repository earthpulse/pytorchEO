from pytorch_eo.metrics.classification import accuracy
from ..BaseTask import BaseTask
import torch

class ImageClassification(BaseTask):

    def __init__(self, model, hparams=None, metrics=None):
        # default hparams
        if hparams is None:
            hparams = {
                'loss': 'CrossEntropyLoss',
                'optimizer': 'Adam'
            }
        # default metrics
        if metrics is None:
            metrics = {'acc': accuracy}
        super().__init__(model, hparams, metrics)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_hat = self(x.to(self.device))
            return torch.softmax(y_hat, axis=1)