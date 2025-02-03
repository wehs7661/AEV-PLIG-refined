import torch
import numpy as np
from math import sqrt
from scipy import stats
from model_defs import GATv2Net

model_dict = {"GATv2Net": GATv2Net}

def get_num_parameters(model):
    """
    counts the number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    """
    function needed for data loaders
    """
    feature_list, protein_seq_list, label_list = [], [], []
    for _features, _protein_seq, _label in batch:
        #print(type(_features), type(_protein_seq), type(_label))
        feature_list.append(_features)
        protein_seq_list.append(_protein_seq)
        label_list.append(_label)
    return torch.Tensor(feature_list), torch.Tensor(protein_seq_list), torch.Tensor(label_list)


def rmse(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the RMSE
    """
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the MSE
    """
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the pearson correlation coefficient
    """
    rp = np.corrcoef(y, f)[0,1]
    return rp


def spearman(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the spearman correlation coefficient
    """
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the concordance index
    """
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
