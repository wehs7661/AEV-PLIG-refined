import sys
import numpy as np
from tqdm import tqdm
from scipy import stats
from rdkit.DataStructs import BulkTanimotoSimilarity


def calc_max_tanimoto_similarity(fps1, fps2):
    """
    Given two sets of fingerprints, calculate the maximum Tanimoto similarity
    between each fingerprint in the first set and all fingerprints in the second set.
    
    Parameters
    ----------
    fps1 : list
        The first set of fingerprints.
    fps2 : list
        The second set of fingerprints.
    
    Returns
    -------
    max_similarities : list
        A list of maximum Tanimoto similarities against the second set of fingerprints for each
        fingerprint in the first set.
    """
    max_similarities = []
    for i in tqdm(range(len(fps1)), file=sys.__stderr__, desc="Calculating Tanimoto similarities"):
        max_similarities.append(BulkTanimotoSimilarity(fps1[i], fps2))
    max_similarities = np.array(max_similarities)
    max_similarities = max_similarities.max(axis=1)

    return max_similarities


def calc_rmse(y_pred, y_true):
    """
    Calculate the root mean square error (RMSE) between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    
    Returns
    -------
    rmse : float
        The RMSE value.
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return rmse


def calc_mse(y_pred, y_true):
    """
    Calculate the mean square error (MSE) between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    
    Returns
    -------
    mse : float
        The MSE value.
    """
    mse = np.mean((y_pred - y_true) ** 2)
    return mse

def calc_pearson(y_pred, y_true):
    """
    Calculate the Pearson correlation coefficient (PCC) between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    
    Returns
    -------
    pcc : float
        The PCC value.
    """
    pcc = np.corrcoef(y_pred, y_true)[0, 1]
    return pcc


def calc_spearman(y_pred, y_true):
    """
    Calculate the Spearman rank correlation coefficient between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    
    Returns
    -------
    spearman : float
        The Spearman rank correlation coefficient.
    """
    spearman = stats.spearmanr(y_pred, y_true)[0]
    return spearman


def calc_kendall(y_pred, y_true):
    """
    Calculate the Kendall rank correlation coefficient between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    
    Returns
    -------
    kendall : float
        The Kendall rank correlation coefficient.
    """
    kendall = stats.kendalltau(y_pred, y_true)[0]
    return kendall


def calc_c_index(y_pred, y_true):
    """
    Calculate the concordance index (C-index) between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    
    Returns
    -------
    c_index : float
        The C-index value.
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
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1

    c_index = S / z
    return c_index

