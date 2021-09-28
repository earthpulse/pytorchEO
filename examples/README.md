# Pytorch EO - Examples

Here you will find multiple examples on how to use Pytorch EO for different tasks.

## Image Classification

The task of image classification consists on assigning one label to an image.

- [EuroSAT](./eurosat.ipynb): Get started with this simple example.
- EuroSAT [advanced](./eurosat_advanced.ipynb): Learn how to leverage the power of Pytorch EO in this advanced example covering topics such as data augmentation, hyperparameter optimization, etc.

## Image Multi-label Classification

The task of image multi-label classification consists on assigning multiple labels to an image.

- [BigEarthNet](./big_earth_net.ipynb): Train an image classifier for a multi-label task.


## Image Segmentation

The task of image segmentation consists on assigning one label to each pixel of an image.

- [LandCoverNet](./land_cover_net.ipynb): Learn how to train a model for image segmentation.

## Data Fusion

Data fusion is a technique that consists on leveraging multiple data sources at the same time to solve a task.

- BigEarthNet [data fusion](./big_earth_net_df.ipynb): Learn how to train an image classifier fusing Sentinel 1 and Sentinel 2 imagery.

## Self-Supervised Learning

Self-Supervised Learning is a technique that consists on training a model without labels.

- EuroSAT [transfer learning](./eurosat_ssl.ipynb) with SSL pre-trained model: In this example we show how to train a model using an SSL pre-trained model from our Models Universe.

## Coming soon ...

- Time series inputs
- Object detection
- Self-Supervised learning (pre-training)
- Multi-task learning
- Multi-modal inputs
- Integrations (models universe, SCAN, SPAI)
- Research (custom datasets, custom tasks)
- Production (export models, upload to universe, access through SPAI)
- Challenges 
- AutoML: NAS, HPO
