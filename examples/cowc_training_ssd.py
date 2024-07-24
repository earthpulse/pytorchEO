from pytorch_eo.datasets.cowc.cowc import Cowc_annot
from pytorch_eo.datasets.cowc.cowc import ObjectDetectionDataModule
import pytorch_lightning as pl

import torch
import torch.optim as optim
from torchvision.models.detection import ssd300_vgg16
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 1. Download data from EOTDL
# 2. put all files you want to use for training/testing into a data folder separated into train/test
#    (e.g. all files from Columbus_CSUAV_AFRL into train and all files from Potsdam_ISPRS into test)
# 3. Use Cowc_annot() function to unpack and split the data into images and annotations 
#    (provide path to data folder containing the test/train folders and ouput folder path)
# cowc_processor = Cowc_annot()
# cowc_processor.process('path/to/folder/data', '/path/to/output/folder')

# set up dataset (provide your '/path/to/output/folder')
data_module = ObjectDetectionDataModule(data_dir='/fastdata/COWC/v1/datasets/ground_truth_sets/processed/', batch_size=4)

data_module.setup()

device = torch.device('cuda') # if torch.cuda.is_available() else torch.device('cpu')

# Initialize SSD model
model = ssd300_vgg16(pretrained=True)
num_classes = 2  # 1 class (car) + background
model.head.classification_head.num_classes = num_classes
model.to(device)

# Train model

val_torch = data_module.val_dataloader()
train_torch = data_module.train_dataloader()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

writer = SummaryWriter()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, targets in train_torch:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(train_torch)
    writer.add_scalar('Loss/train', train_loss, epoch)
    print(f"Epoch: {epoch}, Train Loss: {train_loss}")

    model.eval()
    with torch.no_grad():
        for images, targets in val_torch:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            preds = model(images)

    lr_scheduler.step()

# Save the Model
torch.save(model.state_dict(), 'ssd_model.pth')

# Export model
# Define example inputs
dummy_input = torch.randn(1, 3, 300, 300).to(device)  # Example input tensor of correct shape

# Export to ONNX
torch.onnx.export(model,                     # Model being run
                  dummy_input,               # Example input
                  "ssd_model.onnx",          # Path to save the ONNX model
                  export_params=True,        # Store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # Whether to execute constant folding for optimization
                  input_names = ['input'],   # Input names within the model
                  output_names = ['output'], # Output names within the model
                  dynamic_axes={'input' : {0 : 'batch_size'},    # Variable length axes
                                'output' : {0 : 'batch_size'}})

print('ONNX export success, saved as ssd_model.onnx')