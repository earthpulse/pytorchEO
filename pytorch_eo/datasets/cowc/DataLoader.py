import os
import shutil
import pandas as pd
import lightning as L
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from pytorch_eo.datasets import RGBImageDataset, ConcatDataset
from pytorch_eo.utils import download_eotdl, unzip_file
import xml.etree.ElementTree as ET

###############################################################
## Unfinished downloading and pre-processing of COWC dataset ##
## working but unsatisfactory example in cowc.py 

# Define a function to organize data into train and test folders
def organize_data(data_dir):
    ground_truth_dir = os.path.join(data_dir, 'COWC/v1/datasets/ground_truth_sets')
    data_dir = os.path.join(ground_truth_dir, 'data')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Create train and test directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Define the folders to be used for training and testing
    train_folders = [
        'Columbus_CSUAV_AFRL',
        'Selwyn_LINZ',
        'Utah_AGRC',
        'Vaihingen_ISPRS',
        'Toronto_ISPRS'
    ]
    test_folder = 'Potsdam_ISPRS'

    # Move train folders' content into the train directory
    for folder in train_folders:
        folder_path = os.path.join(ground_truth_dir, folder)
        for file_name in os.listdir(folder_path):
            shutil.move(os.path.join(folder_path, file_name), train_dir)

    # Move test folder's content into the test directory
    test_folder_path = os.path.join(ground_truth_dir, test_folder)
    for file_name in os.listdir(test_folder_path):
        shutil.move(os.path.join(test_folder_path, file_name), test_dir)

# Define the dataset class
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, images, annotations, transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        while idx < len(self.images):
            img_path = self.images[idx]['file_name']
            image = Image.open(img_path).convert("RGB")
            boxes = self.images[idx]['boxes']

            if len(boxes) > 0:  # Only return images with bounding boxes
                if self.transform:
                    image = self.transform(image)

                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.ones((len(boxes),), dtype=torch.int64)  # All labels are 1 (e.g., 'car')

                target = {}
                target["boxes"] = boxes
                target["labels"] = labels

                return image, target
            
            # If no bounding boxes, move to the next index
            idx += 1

    
def parse_voc_annotation(annot_folder, image_folder):
    all_images = []
    all_annotations = []

    for annot_file in os.listdir(annot_folder):
        if annot_file.endswith('.xml'):
            annot_path = os.path.join(annot_folder, annot_file)

            tree = ET.parse(annot_path)
            root = tree.getroot()

            image_info = {}
            image_info['file_name'] = os.path.join(image_folder, root.find('filename').text)
            boxes = []

            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                box = [int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)),
                    int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))]
                boxes.append(box)

            image_info['boxes'] = boxes
            all_images.append(image_info)
            all_annotations.append(boxes)

    return all_images, all_annotations

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

# Define the LightningDataModule
class COWCDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, transform=None, download=False):
        super(COWCDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ])
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = None  # Initialize these later in setup()
        self.val_dataset = None
        self.download = download

    def setup(self, stage=None):
        ground_truth_dir = os.path.join(self.data_dir, 'COWC/v1/datasets/ground_truth_sets')
        train_dir = os.path.join(ground_truth_dir, 'data/train')
        test_dir = os.path.join(ground_truth_dir, 'data/test')

        # Download and unzip the dataset if necessary
        if self.download:
            download_eotdl("COWC", self.data_dir)
            unzip_file(os.path.join(self.data_dir, 'cowc.zip'), self.data_dir)
            organize_data(self.data_dir)

        # Parse annotations
        train_images, train_annotations = parse_voc_annotation(
            os.path.join(train_dir, 'annotations'),
            os.path.join(train_dir, 'images')
        )
        val_images, val_annotations = parse_voc_annotation(
            os.path.join(test_dir, 'annotations'),
            os.path.join(test_dir, 'images')
        )

        self.train_dataset = VOCDataset(train_images, train_annotations, transform=self.train_transforms)
        self.val_dataset = VOCDataset(val_images, val_annotations, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return tuple(zip(*batch))

# Example usage
data_dir = '/home/anna/Desktop'

data_module_2 = COWCDataModule(data_dir=data_dir, batch_size=4, download=True)
data_module_2.setup()
train_loader = data_module_2.train_dataloader()
val_loader = data_module_2.val_dataloader()
