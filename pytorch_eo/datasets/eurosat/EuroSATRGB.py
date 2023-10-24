from .EuroSATBase import EuroSATBase
from pytorch_eo.datasets import RGBImageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EuroSATRGB(EuroSATBase):
    def __init__(
        self,
        batch_size=32,
        download=True,
        path="./data",
        test_size=0.2,
        val_size=0.2,
        train_trans=None,
        val_trans=None,
        test_trans=None,
        num_workers=0,
        pin_memory=False,
        seed=42,
        verbose=False,
        label_ratio=1,
        data_folder="2750",
    ):
        super().__init__(
            batch_size,
            download,
            path,
            test_size,
            val_size,
            train_trans,
            val_trans,
            test_trans,
            num_workers,
            pin_memory,
            seed,
            verbose,
            label_ratio,
        )
        self.url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        self.compressed_data_filename = "EuroSAT.zip"
        self.data_folder = data_folder
        self.in_chans = 3

    def get_image_dataset(self, images):
        return RGBImageDataset(images)

    def setup_trans(self, trans):
        if trans is None:
            return (
                A.Compose(
                    [
                        A.Normalize(0, 1),  # divide by 255
                        ToTensorV2(),  # convert to float tensor and channel first
                    ]
                )
                if trans is None
                else trans
            )
        return trans
