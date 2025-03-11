import tarfile
from tqdm import tqdm


def untar_file(source, path, msg="Extracting"):
    with tarfile.open(name=source) as tar:
        for member in tqdm(
            iterable=tar.getmembers(), total=len(tar.getmembers()), desc=msg
        ):
            tar.extract(path=path, member=member)
