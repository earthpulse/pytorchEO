from pytorch_eo.utils import download_url, unzip_file
from glob import glob
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
from tqdm import tqdm 

def generate_classes_list(uncompressed_data_path) -> list:
    """
    Retrieve classes from the labels associated to the images

    Parameters
    ----------
    uncompressed_data_path : str
        Path to the uncompressed data folder

    Returns
    -------
    classes
        List with the classes
    image_labels
        Dictionary with the path of the images as keys and the labels as values
    """
    images_labels = dict()
    # Get a list with the paths of all the images
    images = glob(f'{uncompressed_data_path}/*/')
    # For each image, get the label, which is the name of the folder containing the image changing 'source' by 'labels'
    # Images only can be FLOODING or NO_FLOODING
    print("Getting the labels of the images...")
    for image in tqdm(images):
        label = get_image_label(image)
        images_labels[image] = label

    return images_labels


def generate_df(images_labels, verbose) -> pd.DataFrame:
    """
    Generate a dataframe with the images and their labels

    Parameters
    ----------
    images_labels : dict
        Dictionary with the path of the images as keys and the labels as values
    verbose : bool
        Whether to print information or not

    Returns
    -------
    df
        Dataframe with the images and their labels
    """
    images, labels = images_labels.keys(), encode_flooding(images_labels.values())
    assert len(images) == len(labels)
    if verbose:
        print(f"Number of images: {len(images)}")
    return pd.DataFrame({"image": images, "label": labels})

def get_image_label(image: str) -> str:
    """
    Get the label of an image
    
    Parameters
    ----------
    image : str
        Path to the image

    Returns
    -------
    label : str
        Label of the image
    """
    image = image.replace('source', 'labels')
    vector_label = os.path.join(image, 'vector_labels.geojson')
    assert os.path.isfile(vector_label), f"Label not found: {vector_label}"
    # Read the label as a geopandas dataframe and get whether the image is FLOODING or NO_FLOODING
    gdf = gpd.read_file(vector_label)
    # Get the label
    flooding = gdf['FLOODING'].values[0]
    return 'FLOODING' if flooding else 'NO_FLOODING'


def encode_flooding(floodings: list) -> list:
    """
    Encode the flooding labels
    
    Parameters
    ----------
    floodings : list
        List with the flooding labels
        
    Returns
    -------
    encoding : list
        List with the encoded flooding labels
    """
    encoding = list()
    for flooding in floodings:
        if flooding == 'FLOODING':
            encoding.append(1)
        elif flooding == 'NO_FLOODING':
            encoding.append(0)
        else:
            raise ValueError(f"Unknown flooding: {flooding}")

    return encoding
