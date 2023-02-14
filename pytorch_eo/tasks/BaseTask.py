import torch
import pytorch_lightning as pl
import torchvision


class BaseTask(pl.LightningModule):
    def __init__(
        self, model, hparams=None, inputs=None, outputs=None, loss_fn=None, metrics=None
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.configure_model(model)
        self.metrics = metrics
        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn

    def configure_model(self, model):
        if isinstance(model, str):
            if not "model" in self.hparams:
                self.hparams["model"] = {}
            model = getattr(torchvision.models, model)(**self.hparams.model)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def compute_metrics(self, y_hat, y):
        return {
            metric_name: metric(y_hat, y)
            for metric_name, metric in self.metrics.items()
        }

    def my_prepare_data(self, batch, keys=None):  # prepare_data already used py pl
        # dict to tuple, in the order given by keys
        # we could keep the dict, but torchscript cannot index dicts
        keys = self.inputs if keys is None else keys
        x = tuple([batch[k] for k in keys])
        return x[0] if len(x) == 1 else x

    def step(self, batch):
        x = self.my_prepare_data(batch, self.inputs)
        y = self.my_prepare_data(batch, self.outputs)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if self.metrics is not None:
            metrics = self.compute_metrics(y_hat, y)
            return loss, metrics
        return loss, {}

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch)
        self.log("loss", loss)
        for metric_name, metric in metrics.items():
            self.log(metric_name, metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, val_metrics = self.step(batch)
        self.log("val_loss", val_loss, prog_bar=True)
        for metric_name, metric in val_metrics.items():
            self.log(f"val_{metric_name}", metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        test_loss, test_metrics = self.step(batch)
        self.log("test_loss", test_loss)
        for metric_name, metric in test_metrics.items():
            self.log(f"test_{metric_name}", metric, prog_bar=True)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x.to(self.device))

    def configure_optimizers(self):
        if not "optim_params" in self.hparams:
            self.hparams["optim_params"] = {}
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams.optim_params
        )
        if "scheduler" in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
                    optimizer, **self.hparams.scheduler_params
                )
            ]
            return [optimizer], schedulers
        return optimizer
