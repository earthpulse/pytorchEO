from pytorch_eo.metrics.classification import accuracy
from ..BaseTask import BaseTask
import torch

class ImageClassification(BaseTask):

    def __init__(self, model, hparams=None, inputs=['image'], outputs=['label'], loss_fn=None, metrics=None):
        
        # defaults
        loss_fn = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        hparams = {'optimizer': 'Adam'} if hparams is None else hparams
        metrics = {'acc': accuracy} if metrics is None else metrics

        super().__init__(model, hparams, inputs, outputs, loss_fn, metrics)

    def compute_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y['label'])

    def compute_metrics(self, y_hat, y):
        return {metric_name: metric(y_hat, y['label']) for metric_name, metric in self.metrics.items()}

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            x = {k: v.to(self.device) for k, v in batch.items() if k in self.inputs}
            y_hat = self(x)
            return torch.softmax(y_hat, axis=1)