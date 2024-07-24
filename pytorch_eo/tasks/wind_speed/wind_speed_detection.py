import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from pytorch_eo.tasks.BaseTask import BaseTask

# Custom RMSE Loss function
class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(nn.MSELoss()(y_pred, y_true))

class WindSpeedDetection(BaseTask):
    def __init__(self, **kwargs):
        self.save_hyperparameters()
        
        self.learning_rate = self.hparams.get("lr", 2e-4)
        self.hidden_size = self.hparams.get("embedding_dim", 50)
        self.dropout = self.hparams.get("dropout", 0.1)
        self.num_outputs = 1  # One prediction for regression

        self.train_step_outputs = []
        self.validation_step_outputs = []

        # Where final model will be saved
        self.output_path = Path.cwd() / self.hparams.get("output_path", "model-outputs")
        self.output_path.mkdir(exist_ok=True)

        # Where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "logs")
        self.log_path.mkdir(exist_ok=True)

        # Prepare the model after setting all necessary attributes
        model = self.prepare_model()
        super(WindSpeedDetection, self).__init__(model=model)

        self.max_epochs = self.hparams.get("max_epochs", 10)
        self.gradient_clip_val = self.hparams.get("gradient_clip_val", 1.0)
        self.val_sanity_checks = self.hparams.get("val_sanity_checks", 0)

    def prepare_model(self):
        embedding_dim = self.hparams.get("embedding_dim", 50)
        dropout = self.hparams.get("dropout", 0.1)
        
        res_model = models.resnet152(pretrained=True)
        res_model.fc = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, self.num_outputs),
        )
        return res_model

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        y = batch["label"].to(self.device)
        criterion = RMSELoss()
        loss = criterion(self.model(x).squeeze(), y.type(torch.FloatTensor).to(self.device))
        self.train_step_outputs.append(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        y = batch["label"].to(self.device)
        criterion = RMSELoss()
        loss = criterion(self.model(x).squeeze(), y.type(torch.FloatTensor).to(self.device))
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_step_outputs).mean()
        self.log("avg_epoch_train_loss", avg_train_loss)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("avg_epoch_val_loss", avg_val_loss)
        self.validation_step_outputs.clear()  # free memory
