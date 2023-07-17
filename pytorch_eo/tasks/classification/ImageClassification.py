from ..BaseTask import BaseTask
import torch
import torchvision
import torchmetrics


class ImageClassification(BaseTask):
    def __init__(
        self,
        model=None,
        hparams=None,
        inputs=["image"],
        outputs=["label"],
        loss_fn=None,
        metrics=None,
        num_classes=None,
    ):
        # defaults
        if num_classes is None and model is None:
            raise ValueError("num_classes or model must be provided")
        if model is None:
            model = torchvision.models.resnet18()
            model.fc = torch.nn.Linear(512, num_classes)
        loss_fn = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        hparams = {"optimizer": "Adam"} if hparams is None else hparams
        if metrics is None and num_classes is None:
            raise ValueError("num_classes must be provided if metrics is None")
        metrics = (
            {"acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)}
            if metrics is None
            else metrics
        )
        super().__init__(model, hparams, inputs, outputs, loss_fn, metrics)

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            x = self.my_prepare_data(batch)
            y_hat = self(x)
            return torch.softmax(y_hat, axis=1)
