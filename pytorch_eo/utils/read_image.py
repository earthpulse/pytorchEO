from skimage import io


def read_image(src):
    return io.imread(src)


def read_ms_image(src, bands=None):
    if bands:
        return io.imread(src)[..., bands]
    return read_image(src)
