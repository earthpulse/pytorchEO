import zipfile
from tqdm import tqdm


def unzip_file(source, path, msg="Extracting"):
    with zipfile.ZipFile(source) as zf:
        for member in tqdm(zf.infolist(), desc=msg):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                pass
