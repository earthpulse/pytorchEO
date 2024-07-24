import lightning.pytorch as pl  # changed from pytorch_lightning
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

#####################################################
### Not working yet in test Notebook  #######


# Dataset class to handle the data loading
class WindDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['file_name']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]['Wind Speed']
        return {"image": image, "label": label}

# DataModule to handle data preprocessing and loading
class WindData(pl.LightningDataModule):
    def __init__(
        self, 
        data_path, 
        batch_size=10, 
        num_workers=0,
        train_transform=None, 
        val_transform=None, 
        test_transform=None
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

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

    def setup(self, stage=None):
        self.prepare_data()

        self.train_dataset = WindDataset(self.train_data, transform=self.train_transform)
        self.val_dataset = WindDataset(self.val_data, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
