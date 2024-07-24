
import torch
import pytorch_lightning as pl
from pytorch_eo.datasets.cyclone.tropical_cyclone import WindDataModule
from pytorch_eo.datasets.cyclone.tropical_cyclone import WindModel
from PIL import Image

##################################################################
## Exact same code as in Notebook but getting Attribute Error ####


hparams = {
    "lr": 2e-4,
    "embedding_dim": 100,
    "dropout": 0.1,
    "max_epochs": 4,
    "batch_size": 10,
    "num_workers": 0,
    "gradient_clip_val": 1,
    "val_sanity_checks": 0,
    "output_path": "windspeed-model-outputs",
    "log_path": "logs-windspeed",
}

# Data path
data_path = '/home/anna/.cache/eotdl/datasets/tropical-cyclone-dataset/v1' # train/test tar files must be unzipped in separate train/test folders

# Create data module
data_module = WindDataModule(data_path=data_path, batch_size=hparams['batch_size'], num_workers=hparams['num_workers'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WindModel(**hparams).to(device)

# Train the model
trainer = pl.Trainer(
    max_epochs=hparams['max_epochs'],
    default_root_dir=hparams['output_path'],
    logger=pl.loggers.TensorBoardLogger(save_dir=hparams['log_path'], name="wind_speed_model"),
    callbacks=[pl.callbacks.ModelCheckpoint(
        dirpath=hparams['output_path'],
        monitor="avg_epoch_val_loss",
        mode="min",
        verbose=True,
    )],
    gradient_clip_val=hparams['gradient_clip_val'],
    num_sanity_val_steps=hparams['val_sanity_checks'],
)
trainer.fit(model, datamodule=data_module)