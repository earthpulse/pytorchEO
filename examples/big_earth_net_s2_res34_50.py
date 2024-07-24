import argparse
import timm
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import average_precision_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from einops import rearrange
from pytorch_eo.datasets.big_earth_net import BigEarthNet
from pytorch_eo.datasets.big_earth_net.utils import LABELS19
from pytorch_eo.datasets.sensors import Sensors, S2
from pytorch_eo.tasks.classification import ImageMultilabelClassification
import torch

## Run with: python .../filename_of_your_script.py --backbone resnet50/resnet34

# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description="BigEarthNet Classification Script")
    parser.add_argument('--backbone', type=str, choices=['resnet34', 'resnet50'], required=True, help='Model backbone to use: "resnet34" or "resnet50"')
    return parser.parse_args()

# Main function
def main():
    args = get_args()
    
    # dataset
    def clip(x, **kwargs): return np.clip(x / 4000, 0., 1.)
    def add_channel(x, **kwargs): return rearrange(x, 'h w -> h w 1') if x.ndim == 2 else x

    train_trans = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Transpose(),
        A.RandomRotate90(),
        A.Lambda(image=clip),
        A.Lambda(image=add_channel),
        ToTensorV2(),
    ])

    val_trans = A.Compose([
        A.Lambda(image=clip),
        A.Lambda(image=add_channel),
        ToTensorV2(),
    ])

    ds = BigEarthNet(
        path="/fastdata/BigEarthNet", 
        batch_size=64, 
        bands={Sensors.S2: [S2.red, S2.green, S2.blue]},
        label_groups=LABELS19,
        train_trans=train_trans, 
        val_trans=val_trans,
        num_workers=8,
        pin_memory=True,
    )

    ds.setup()

    # hyperparameters
    hparams = {
        'optimizer': 'Adam',
        'optim_params': {
            'lr': 3e-4,
        }
    }

    # model
    model = timm.create_model(args.backbone, pretrained=True, num_classes=ds.num_classes, in_chans=ds.in_chans)

    # train 
    def my_map(y_hat, y):
        return average_precision_score(y.cpu(), torch.sigmoid(y_hat).detach().cpu(), average='micro')

    task = ImageMultilabelClassification(model, hparams=hparams, metrics={'map': my_map})

    torch.set_float32_matmul_precision('high')  # more perf on some GPUs

    trainer = L.Trainer(
        devices=1,
        accelerator='cuda',
        precision="16-mixed",
        max_epochs=10,
        limit_train_batches=100,
        limit_val_batches=100,
        logger=WandbLogger(project="bigearthnet-classification-S2", name=args.backbone),
        callbacks=[
            ModelCheckpoint(
                monitor="val_map",
                mode="max",
                save_top_k=1,
                dirpath="checkpoints",
                filename=f"{args.backbone}-{{epoch:02d}}-{{val_map:.4f}}-{{S2}}",
            )
        ],
    )

    trainer.fit(task, ds)

if __name__ == "__main__":
    main()
