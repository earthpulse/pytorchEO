import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp


class ImageSegmentation(pl.LightningModule):
    def __init__(self, model, backbone, in_chans, num_classes, pretrained=None, optimizer='Adam', lr=1e-3, scheduler=None, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = getattr(smp, model)(
            encoder_name=backbone,
            encoder_weights=pretrained,
            in_channels=in_chans,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = self(x.to(self.device))
            return torch.sigmoid(preds)

    def iou(self, pr, gt, th=0.5, eps=1e-7):
        pr = torch.sigmoid(pr) > th
        gt = gt > th
        intersection = torch.sum(gt * pr, axis=(-2, -1))
        union = torch.sum(gt, axis=(-2, -1)) + torch.sum(pr,
                                                         axis=(-2, -1)) - intersection + eps
        ious = (intersection + eps) / union
        return torch.mean(ious)

    def step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        iou = self.iou(y_hat, y)
        return loss, iou

    def training_step(self, batch, batch_idx):
        loss, iou = self.step(batch)
        self.log('loss', loss)
        self.log('iou', iou, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, iou = self.step(batch)
        self.log('test_loss', loss)
        self.log('test_iou', iou)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(
                    optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers
        return optimizer
