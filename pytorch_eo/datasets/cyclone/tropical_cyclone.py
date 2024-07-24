import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from PIL import Image as pil_image
from PIL import Image
import os
import shutil
from pytorch_eo.utils import download_eotdl, unzip_file, untar_file
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

##########################################################################
### When downloading the data from EOTDL, the .csv files are not present - 
# important for model so I downloaded them from source cooperative and add them to the folder with this code:
# azcopy sync https://radiantearth.blob.core.windows.net/mlhub/nasa-tropical-storm-challenge . --recursive=false

##########################################################################

# Custom RMSELoss for training
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# Dataset class to handle the data loading
class WindDataset(Dataset):
    def __init__(self, data, transform=None, is_test=False):
        self.data = data
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['file_name']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return {"image": image, "idx": idx}
        
        label = self.data.iloc[idx]['Wind Speed']
        return {"image": image, "label": label}

# DataModule to handle data preprocessing and loading
class WindDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=10, num_workers=0):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        # Read metadata and labels
        train_metadata = pd.read_csv(self.data_path / "training_set_features.csv")
        train_labels = pd.read_csv(self.data_path / "training_set_labels.csv")

        # Merge metadata and labels
        full_metadata = train_metadata.merge(train_labels, on="Image ID")

        # Create file paths
        full_metadata["file_name"] = full_metadata["Image ID"].apply(self.create_file_path)

        # Split data
        images_per_storm = full_metadata.groupby("Storm ID").size().to_frame("images_per_storm")
        full_metadata = full_metadata.merge(images_per_storm, how="left", on="Storm ID")
        full_metadata["pct_of_storm"] = full_metadata.groupby("Storm ID").cumcount() / full_metadata.images_per_storm
        train_data = full_metadata[full_metadata.pct_of_storm < 0.8].drop(["images_per_storm", "pct_of_storm"], axis=1)
        val_data = full_metadata[full_metadata.pct_of_storm >= 0.8].drop(["images_per_storm", "pct_of_storm"], axis=1)

        # Sample data for performance
        self.train_data = train_data.sample(frac=0.1, replace=False, random_state=1)
        self.val_data = val_data.sample(frac=0.1, replace=False, random_state=1)

    def create_file_path(self, image_id):
        return self.data_path / 'train' / 'nasa_tropical_storm_competition_train_source' / f'nasa_tropical_storm_competition_train_source_{image_id}' / "image.jpg"

    def train_dataloader(self):
        train_data = WindDataset(self.train_data, transform=self.transform)
        return DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        val_data = WindDataset(self.val_data, transform=self.transform)
        return DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

# Lightning module for the model
class WindModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(WindModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = self.hparams.get("lr", 2e-4)
        self.hidden_size = self.hparams.get("embedding_dim", 50)
        self.dropout = self.hparams.get("dropout", 0.1)
        self.max_epochs = self.hparams.get("max_epochs", 1)
        self.num_workers = self.hparams.get("num_workers", 0)
        self.batch_size = self.hparams.get("batch_size", 10)
        self.num_outputs = 1  # One prediction for regression

        self.train_step_outputs = []
        self.validation_step_outputs = []

        # Where final model will be saved
        self.output_path = Path.cwd() / self.hparams.get("output_path", "model-outputs")
        self.output_path.mkdir(exist_ok=True)

        # Where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "logs")
        self.log_path.mkdir(exist_ok=True)

        self.model = self.prepare_model()

    def prepare_model(self):
        res_model = models.resnet152(pretrained=True)
        res_model.fc = nn.Sequential(
            nn.Linear(2048, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.num_outputs),
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

    def fit(self):
        logger = pl.loggers.TensorBoardLogger(save_dir=self.log_path, name="benchmark_model")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="avg_epoch_val_loss",
            mode="min",
            verbose=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            default_root_dir=self.output_path,
            logger=logger,
            callbacks=[checkpoint_callback],
            gradient_clip_val=self.hparams.get("gradient_clip_val", 1),
            num_sanity_val_steps=self.hparams.get("val_sanity_checks", 0),
        )
        self.trainer.fit(self)

    @torch.no_grad()
    def make_submission_frame(self, x_test):
        test_dataset = WindDataset(x_test, is_test=True)
        test_dataloader = DataLoader(test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)
        submission_frame = pd.DataFrame(index=x_test['Image ID'], columns=["Wind Speed"])
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            x = batch["image"].to(self.device)
            preds = self(x)
            submission_frame.loc[batch["Image ID"], "Wind Speed"] = preds.detach().cpu().numpy().squeeze()
        submission_frame['Wind Speed'] = submission_frame['Wind Speed'].astype(float)
        return submission_frame