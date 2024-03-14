try:
    import eotdl
except ImportError:
    raise ImportError(
        "eotdl not found. Please install it by running 'pip install eotdl'"
    )


def download_eotdl(dataset_name, path, force=True):
    from eotdl.datasets import download_dataset

    print(f"Downloading {dataset_name} to {path}...")
    return download_dataset(dataset_name, path=path, force=force)
