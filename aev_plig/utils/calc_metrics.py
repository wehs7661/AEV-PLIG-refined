import sys
import numpy as np
from tqdm import tqdm
from numba import njit
from scipy import stats
from rdkit.DataStructs import BulkTanimotoSimilarity


def calc_weighted_mean(values, weights):
    """
    Calculate the weighted mean of a set of values.
    
    Parameters
    ----------
    values : np.ndarray
        The values to calculate the weighted mean for.
    weights : np.ndarray
        The weights corresponding to each value.
    
    Returns
    -------
    weighted_mean : float
        The weighted mean of the values.
    """
    weighted_mean = np.sum(values * weights) / np.sum(weights)
    return weighted_mean


def calc_bootstrap_uncertainty(values, weights, n_iterations=10000):
    """
    Calculate the uncertainty of a weighted mean using bootstrapping.
    
    Parameters
    ----------
    values : np.ndarray
        The values to calculate the uncertainty for.
    weights : np.ndarray
        The weights corresponding to each value.
    n_iterations : int, optional
        The number of bootstrap iterations to perform. Default is 10000.
    
    Returns
    -------
    ci_bounds : list
        A list containing the lower and upper bounds of the 95% confidence interval for the weighted mean.
    """
    n = len(values)
    bootstrap_means = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        bootstrap_values = values[indices]
        bootstrap_weights = weights[indices]
        bootstrap_means[i] = calc_weighted_mean(bootstrap_values, bootstrap_weights)
    
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    ci_bounds = (lower_bound, upper_bound)
    
    return ci_bounds


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


def calc_pearson(y_pred, y_true, groups=None, n_min=10):
    """
    Calculate the Pearson correlation coefficient (PCC) between two sets of values.
    
    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y_true : np.ndarray
        The true values.
    groups : list of str, optional
        A list of groups that correspond to each prediction/true value pair. If provided, the PCC will
        be calculated for each group separately. And a weighted average of the PCCs will be returned
        with uncertainty estimated by bootstrapping (half the 95% confidence interval).
    n_min : int, optional
        The minimum number of samples required in each group to calculate the PCC. If a group has fewer
        samples than this, it will be skipped. Default is 10. This parameter is ignored if groups are not provided.
    
    Returns
    -------
    pcc : float
        The PCC value. If groups are provided, it will be the weighted average of the PCCs for each group.
    uncertainty : float
        The uncertainty of the weighted average PCC, estimated by bootstrapping. Only returned if groups are provided.
    pcc_dict : dict
        A dictionary with group names as keys and their corresponding PCC values as values. Only returned if groups are provided.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if groups is not None:
        groups = np.array(groups)
        unique_groups = np.unique(groups)
        pcc_dict = {}
        pcc_values, group_sizes = [], []
        for group in unique_groups:
            if np.sum(groups == group) < n_min:
                continue
            mask = groups == group
            pcc = np.corrcoef(y_pred[mask], y_true[mask])[0, 1]
            
            pcc_values.append(pcc)
            group_sizes.append(np.sum(mask))
            pcc_dict[group] = pcc
        
        # Calculate weighted average PCC
        pcc = calc_weighted_mean(np.array(pcc_values), np.array(group_sizes))

        # Calculate uncertainty using bootstrapping
        ci_bounds = calc_bootstrap_uncertainty(np.array(pcc_values), np.array(group_sizes))
        uncertainty = (ci_bounds[1] - ci_bounds[0]) / 2
    else:
        pcc = np.corrcoef(y_pred, y_true)[0, 1]
        uncertainty = None
        pcc_dict = None

    return pcc, uncertainty, pcc_dict

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

