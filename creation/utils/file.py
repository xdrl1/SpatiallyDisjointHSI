import h5py
import numpy
import rasterio


def LoadTiff(path):
    with rasterio.open(path) as tiff:
        raw = tiff.read()
        res = numpy.array(raw)
        if res.shape[0] == 1:
            return res[0, :, :]
        return res
