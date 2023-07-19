from pytorch_eo.utils import download_url, unzip_file
from glob import glob
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio


def download_data(path, compressed_data_filename, data_folder, download, url, verbose):
    # download data
    compressed_data_path = path / compressed_data_filename
    uncompressed_data_path = path / data_folder
    if download:
        # create data folder
        os.makedirs(path, exist_ok=True)
        # extract
        if not os.path.isdir(uncompressed_data_path):
            # check data is not already downloaded
            if not os.path.isfile(compressed_data_path):
                print("downloading data ...")
                download_url(url, compressed_data_path)
            else:
                print("data already downloaded !")

            unzip_file(compressed_data_path, path, msg="extracting data ...")
        else:
            if verbose:
                print("data already extracted !")
            # TODO: check data is correct
    else:
        assert os.path.isdir(uncompressed_data_path), "data not found"
        # TODO: check data is correct
    return uncompressed_data_path


def generate_classes_list(uncompressed_data_path):
    # retrieve classes from the labels associated to the imagesç
    classes = list()
    images_labels = dict()
    # Get a list with the paths of all the mosaic images
    images = glob(f'{uncompressed_data_path}/**/mosaic.tif', recursive=True)
    # For each image, get the label, which is the name of the folder containing the image changing 'source' by 'labels'
    # Images only can be FLOODING or NO_FLOODING
    for image in images:
        label = get_image_label(image)
        if label not in classes:
            classes.append(label)
        images_labels[image] = label

    return classes, images_labels


def generate_df(images_labels, verbose):
    images, labels = images_labels.keys(), encode_flooding(images_labels.values())
    assert len(images) == len(labels)
    if verbose:
        print(f"Number of images: {len(images)}")
    return pd.DataFrame({"image": images, "label": labels})


def mosaic_images(uncompressed_data_path: str, verbose: bool = False):
    mosaic_done = False
    # Get a list with the paths of all the images
    images = glob(f'{uncompressed_data_path}/**/*.tif', recursive=True)
    # Get a list with all the directories containing images
    raster_dirs = list(set([os.path.dirname(image) for image in images]))
    # For each directory, mosaic the images
    for raster_dir in raster_dirs:
        # Get the path of all the images in the directory
        images_to_mosaic = glob(f'{raster_dir}/*.tif')
        # Sort the images
        images_to_mosaic.sort()
        if os.path.join(raster_dir, 'mosaic.tif') in images_to_mosaic:
            ds = rio.open(os.path.join(raster_dir, 'mosaic.tif'))
            continue
        # Set the path of the mosaic image
        mosaic_image = os.path.join(raster_dir, 'mosaic.tif')
        # Merge
        src_to_mosaic = list()
        for fp in images_to_mosaic:
            src = rio.open(fp)
            src_to_mosaic.append(src)
        # Prepara the images
        imgs = [src.read(1) for src in src_to_mosaic]
        # Copy the metadata
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update(
            {"driver": "GTiff",
            "transform": src.transform,
            "crs": "EPSG:4326",
            "count": len(imgs)
            }
        )
        # TODO resize to the biggest image
        # Write the mosaic image
        idx = 1
        with rio.open(mosaic_image, "w", **out_meta) as dest:
            mosaic_done = True if not mosaic_done else mosaic_done
            for img in imgs:
                dest.write(img, idx)
                idx += 1
    if verbose and mosaic_done:
        print(f"Mosaic images created")


def remove_mosaics(uncompressed_data_path: str):
    # Get a list with the paths of all the mosaic images
    images = glob(f'{uncompressed_data_path}/**/mosaic.tif', recursive=True)
    # Remove all the mosaic images
    for image in images:
        os.remove(image)


def get_image_label(image: str):
    raster_dir = os.path.dirname(image)
    label = raster_dir.replace('source', 'labels')
    vector_label = os.path.join(label, 'vector_labels.geojson')
    assert os.path.isfile(vector_label), f"Label not found: {vector_label}"
    # Read the label as a geopandas dataframe and get whether the image is FLOODING or NO_FLOODING
    gdf = gpd.read_file(vector_label)
    # Get the label
    flooding = gdf['FLOODING'].values[0]

    return 'FLOODING' if flooding else 'NO_FLOODING'


def encode_flooding(floodings: list) -> list:
    encoding = list()
    for flooding in floodings:
        if flooding == 'FLOODING':
            encoding.append(1)
        elif flooding == 'NO_FLOODING':
            encoding.append(0)
        else:
            raise ValueError(f"Unknown flooding: {flooding}")

    return encoding
