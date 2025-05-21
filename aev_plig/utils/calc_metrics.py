import sys
import numpy as np
from tqdm import tqdm
from numba import njit
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
    Calculate the Kendall's tau correlation coefficient between two sets of values.
    
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


@njit
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
    ind = np.argsort(y_true)
    y = y_true[ind]
    f = y_pred[ind]
    n = len(y)
    S = 0.0
    z = 0.0

    for i in range(n):
        for j in range(i):
            if y[i] > y[j]:
                z += 1
                if f[i] > f[j]:
                    S += 1
                elif f[i] == f[j]:
                    S += 0.5
    return S / z if z > 0 else 0.0

