import timm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# TODO: esto hay que modularizarlo bien :)


class ImageClassifier(pl.LightningModule):

    def __init__(self, backbone, pretrained, in_chans, num_classes, optimizer='Adam', lr=1e-3, scheduler=None, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def accuracy(self, y_hat, y):
        return (torch.argmax(y_hat, axis=1) == y).sum() / y.shape[0]

    def step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, val_acc = self.step(batch)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        test_loss, test_acc = self.step(batch)
        self.log('test_acc', test_acc)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_hat = self(x.to(self.device))
            return torch.softmax(y_hat, axis=1)

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
