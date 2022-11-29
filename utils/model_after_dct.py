"""Models of PyTorch version."""
import torch
import torch.fft
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
import numpy as np


class DCT30x30(nn.Module):
    def __init__(self, filter_path, div_255=False) -> None:
        super().__init__()
        w = np.expand_dims(np.load(filter_path), 1)
        if div_255:
            w /= 255
        # w = np.swapaxes(w, -1, -2)
        state = {'weight': torch.from_numpy(w).float()}
        self.kernel = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=30, stride=30,
            padding=0, bias=False
        )
        self.kernel.load_state_dict(state)

    def forward(self, x):
        return self.kernel(x)

class NetWithDCTConv(nn.Module):
    def __init__(self, dct_conv: DCT30x30, net: nn.Module) -> None:
        super().__init__()
        self.dct_conv = dct_conv.eval()
        self.net = net
        for p in self.dct_conv.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        out = self.dct_conv(x)
        out = self.net(out)
        return out

def cutblock(img, block_size, block_dim):
    blockarray=[]
    for i in range(0, block_dim):
        blockarray.append([])
        for j in range(0, block_dim):
            blockarray[i].append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])
    return np.asarray(blockarray)

def dct_torch(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    if torch.__version__ < '1.8' and True:
        Vc = torch.rfft(v, 1, onesided=False)
    else:
        # FIXME: inconsistent with pytorch version < 1.8
        Vc = torch.fft.rfft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V

#calculate 2D DCT of a matrix
def dct2_torch(img):
    return dct_torch(dct_torch(img.T, norm='ortho').T, norm='ortho')

def zigzag_torch(t, fealen):
    if fealen != 32:
        t_idx = torch.empty(fealen, device=t.device, dtype=torch.long)
        idx = 0
        for i in range(fealen):
            if idx >= fealen:
                break
            elif i == 0:
                t_idx[0] = 0
                idx=idx+1
            elif i%2==1:
                for j in range(0, i+1):
                    if idx<fealen:
                        t_idx[idx] = j * t.size(0) + i - j
                        idx=idx+1
                    else:
                        break
            elif i%2==0:
                for j in range(0, i+1):
                    if idx<fealen:
                        t_idx[idx] = (i - j) * t.size(0) + j
                        idx=idx+1
                    else:
                        break
    else:
        # for fealen == 32, just use:
        t_idx = torch.tensor([0, 1, 128, 256, 129, 2, 3, 130, 257, 384, 512, 385, 258, 131, 4, 5, 132, 259, 386, 513, 640, 768, 641, 514, 387, 260, 133, 6, 7, 134, 261, 388],
                             dtype=torch.long, device=t.device)
    tt = torch.index_select(t.flatten(), dim=0, index=t_idx)
    return tt


def subfeature_torch(imgraw, fealen):
    if fealen > len(imgraw) ** 2:
        print ('ERROR: Feature vector length exceeds block size.')
        print ('Abort.')
        quit()
    imgraw = torch.from_numpy(imgraw)#.cuda()
    scaled = dct2_torch(imgraw)
    feature = zigzag_torch(scaled, fealen)
    return feature

# Generate DCT from image
def feature(img, block_size, block_dim, fealen):
    img = img / 255
    feaarray = np.empty(fealen*block_dim*block_dim).reshape(fealen, block_dim, block_dim)
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            if (blocked[i, j].max() == 0):
                feaarray[:,i,j] = 0
                continue
            # featemp=subfeature(blocked[i,j], fealen)
            featemp=subfeature_torch(blocked[i,j], fealen)
            feaarray[:,i,j]=featemp
    return feaarray

# Generate DCT from image
def feature_torch(img, block_size, block_dim, fealen):
    img = img / 255
    feaarray = torch.empty((fealen, block_dim, block_dim))
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            if blocked[i, j].max() == 0:
                feaarray[:,i,j] = 0
                continue
            featemp=subfeature_torch(blocked[i,j], fealen)
            feaarray[:,i,j]=featemp
    return feaarray


if __name__ == '__main__':
    def make_dct_conv():
        from tqdm import trange
        w = torch.zeros(32, 30, 30)
        for i in trange(110):
            for j in range(110):
                v = np.zeros((110,) * 2, dtype=np.long)
                v[i, j] = 255
                vv = feature_torch(v, 110, 1, 32)
                w[:, i, j] = vv.flatten()
        np.save('mydct_conv_30.npy', w.numpy())
    make_dct_conv()
    dct30=DCT30x30('mydct_conv_30.npy')
    model=NetWithDCTConv(DCT30x30, target_model)