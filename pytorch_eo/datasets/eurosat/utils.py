from pytorch_eo.utils import download_url, unzip_file
import os
import pandas as pd


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
    # retrieve classes from folder structure
    return sorted(os.listdir(uncompressed_data_path))


def generate_df(classes, uncompressed_data_path, verbose):
    images, labels = [], []
    for ix, label in enumerate(classes):
        _images = os.listdir(uncompressed_data_path / label)
        images += [str(uncompressed_data_path / label / img) for img in _images]
        labels += [ix] * len(_images)
    assert len(images) == len(labels)
    if verbose:
        print(f"Number of images: {len(images)}")
    return pd.DataFrame({"image": images, "label": labels})
