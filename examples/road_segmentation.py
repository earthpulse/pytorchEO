import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_eo.tasks import ImageSegmentation
from pytorch_eo.datasets import DeepGlobeRoadExtraction
import onnxruntime
import numpy as np

# dataset


def norm(x, **kwargs):
    return (x / 255.0).astype("float32")


def to_grey(x, **kwargs):
    return x[:1, ...]


train_trans = A.Compose(
    [
        # flips
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Transpose(),
        # rotate
        A.RandomRotate90(),
        # normalize
        A.Lambda(image=norm, mask=norm),
        ToTensorV2(transpose_mask=True),
        A.Lambda(mask=to_grey),
    ]
)

val_trans = A.Compose(
    [
        A.Lambda(image=norm, mask=norm),
        ToTensorV2(transpose_mask=True),
        A.Lambda(mask=to_grey),
    ]
)

ds = DeepGlobeRoadExtraction(
    batch_size=8,  # adjust to GPU memory
    num_workers=20,  # adjust to CPU cores
    pin_memory=True,
)


# hyperparameters

hparams = {
    "optimizer": "Adam",
    "optim_params": {
        "lr": 3e-4,
    },
}

# model


model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # imagenet
    in_channels=3,
    classes=1,
)

# train

task = ImageSegmentation(
    model,
    hparams,
    num_classes=ds.num_classes,
)

trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=10,
    logger=WandbLogger(project="deepglobe-road-extraction", name="unet-resnet50"),
    callbacks=[
        ModelCheckpoint(
            monitor="val_iou",
            mode="max",
            save_top_k=1,
            dirpath="checkpoints",
            filename="unet-resnet50-{epoch:02d}-{val_iou:.2f}",
        )
    ],
)

trainer.fit(task, ds)
