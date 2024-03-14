import torch
from pytorch_eo.metrics.segmentation import iou
from ..BaseTask import BaseTask

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation_models_pytorch is not installed. "
        "Please install it with `pip install segmentation_models_pytorch`"
    )


class ImageSegmentation(BaseTask):
    def __init__(
        self,
        model=None,
        hparams=None,
        inputs=["image"],
        outputs=["mask"],
        loss_fn=None,
        metrics=None,
        num_classes=None,
    ):
        # defaults
        if num_classes is None and model is None:
            raise ValueError("num_classes or model must be provided")
        if model is None:
            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",  # imagenet
                in_channels=3,
                classes=num_classes,
            )
        loss_fn = torch.nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn
        hparams = {"optimizer": "Adam"} if hparams is None else hparams
        if metrics is None and num_classes is None:
            raise ValueError("num_classes must be provided if metrics is None")
        metrics = {"iou": iou} if metrics is None else metrics
        super().__init__(model, hparams, inputs, outputs, loss_fn, metrics)

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            x = self.my_prepare_data(batch)
            y_hat = self(x)
            return torch.sigmoid(y_hat)
