import h5py
import numpy
import torch


class HSIClsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(HSIClsDataset, self).__init__()
        self.x, self.y = torch.from_numpy(x), torch.from_numpy(y)
        self.x = self.x.to(torch.float32)
        self.y = self.y.type(torch.LongTensor)
        self.y -= 1
        self.nCls = len(numpy.unique(self.y))

    def GetNCHL(self):
        return self.x.shape[1]

    def GetNCls(self):
        return self.nCls

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def ZScoreNorm(img):
    return (img - img.mean(axis=(1,2), keepdims=True)) / img.std(axis=(1,2), keepdims=True)


def GetPatch(rawData, patchSize, norm=True):
    assert patchSize % 2 == 1
    nPad = int((patchSize-1)/2)
    rawX, rawY, trMask, teMask = rawData
    c, h, w = rawX.shape
    # norm
    if norm:
        rawX = ZScoreNorm(rawX)
    # pad
    padX = numpy.zeros((c, h+patchSize-1, w+patchSize-1))
    padX[:, nPad:nPad+h, nPad:nPad+w] = rawX
    # prepare return
    trX = numpy.zeros((trMask.sum(), c, patchSize, patchSize))
    trY = numpy.zeros((trMask.sum(), ))
    teX = numpy.zeros((teMask.sum(), c, patchSize, patchSize))
    teY = numpy.zeros((teMask.sum(), ))
    m, n = 0, 0
    for i in range(h):
        for j in range(w):
            if trMask[i][j]:
                # for training
                trX[m] = padX[:, i:i+patchSize, j:j+patchSize]
                trY[m] = rawY[i][j]
                m += 1
            if teMask[i][j]:
                # for testing
                teX[n] = padX[:, i:i+patchSize, j:j+patchSize]
                teY[n] = rawY[i][j]
                n += 1
    assert m == trMask.sum()
    assert n == teMask.sum()
    assert trY.min() > 0
    assert teY.min() > 0
    return trX, trY, teX, teY


def LoadDataset(path, patchSize=11, norm=True):
    with h5py.File(path, 'r') as h5r:
        # TODO: Here triggers a warning due to h5py library, which is not a big deal
        # `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rawX = numpy.array(h5r.get('HSI'))
        rawY = numpy.array(h5r.get('gt'))
        trMask = numpy.array(h5r.get('trMask'))
        vaMask = numpy.array(h5r.get('vaMask'))
        trX, trY, vaX, vaY = GetPatch((rawX, rawY, trMask, vaMask), patchSize=patchSize, norm=norm)
        return HSIClsDataset(trX, trY), HSIClsDataset(vaX, vaY)


if __name__ == '__main__':
    # trSet, vaSet = LoadDataset('IndianPines.h5')
    trSet, vaSet = LoadDataset('Salinas.h5')
    # trSet, vaSet = LoadDataset('UniTrento.h5')
    print(trSet.GetNCHL(), trSet.GetNCls())
    print(vaSet.GetNCHL(), vaSet.GetNCls())
    trLoader = torch.utils.data.DataLoader(trSet, batch_size=64, shuffle=True)
    vaLoader = torch.utils.data.DataLoader(vaSet, batch_size=64, shuffle=False)
    for x, y in trLoader:
        print(x.shape, y.shape)
        break
    for x, y in vaLoader:
        print(x.shape, y.shape)
        break