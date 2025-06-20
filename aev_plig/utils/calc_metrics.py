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

class MetricCalculator:
    """
    A class to calculate various metrics between predicted and true values, including
    Pearson correlation, Spearman correlation, Kendall correlation, C-index, RMSE, and MSE.    
    """
    def __init__(self, y_pred, y_true, groups=None, n_min=10):
        """
        Initializes the MetricCalculator class.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted values.
        y_true : np.ndarray
            The true values.
        groups : list of str, optional
            A list of groups that correspond to each prediction/true value pair. If provided, the metrics will
            be calculated for each group separately. And a weighted average of the metrics will be returned
            with uncertainty estimated by bootstrapping (half the 95% confidence interval).
        n_min : int, optional
            The minimum number of samples required in each group to calculate the metrics. If a group has fewer
            samples than this, it will be skipped. Default is 10. This parameter is ignored if groups are not provided.
        """
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)
        self.groups = np.array(groups) if groups is not None else None
        self.n_min = n_min

        if len(self.y_pred) != len(self.y_true):
            raise ValueError("y_pred and y_true must have the same length.")
        
        if self.groups is not None and len(self.y_pred) != len(self.groups):
            raise ValueError("y_pred and groups must have the same length.")

    @staticmethod
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

    @staticmethod
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
            bootstrap_means[i] = MetricCalculator.calc_weighted_mean(bootstrap_values, bootstrap_weights)
        
        lower_bound = np.percentile(bootstrap_means, 2.5)
        upper_bound = np.percentile(bootstrap_means, 97.5)
        ci_bounds = (lower_bound, upper_bound)
        
        return ci_bounds

    def _metric_by_group(self, metric_func):
        """
        Calculate the metric of interest (e.g., Pearson, Spearman, Kendall, MSE, and RMSE) for each group.
        
        Parameters
        ----------
        metric_func : callable
            The correlation function to apply (e.g., stats.pearsonr, stats.spearmanr, stats.kendalltau).
            Note that the function should return a tuple where the first element is the correlation value.

        Returns
        -------
        metric : float
            The calculated correlation. If groups are provided, it will be the weighted average of the correlations for each group.
        uncertainty : float
            The uncertainty of the weighted average correlation, estimated by bootstrapping. Only returned if groups are provided.
        metric_dict : dict
            A dictionary with group names as keys and their corresponding correlation values as values. Only returned if groups are provided.
        """
        if self.groups is not None:
            unique_groups = np.unique(self.groups)
            metric_dict = {}
            metric_values, group_sizes = [], []
            for group in unique_groups:
                if np.sum(self.groups == group) < self.n_min:
                    continue
                mask = self.groups == group
                metric_value = metric_func(self.y_pred[mask], self.y_true[mask])[0]
                
                metric_values.append(metric_value)
                group_sizes.append(np.sum(mask))
                metric_dict[group] = metric_value
            
            # If no groups meet the minimum size requirement, return None
            if not metric_values:
                print("No groups meet the minimum size requirement.")
                return None, None, None

            # Calculate weighted average correlation
            metric = self.calc_weighted_mean(np.array(metric_values), np.array(group_sizes))

            # Calculate uncertainty using bootstrapping
            ci_bounds = self.calc_bootstrap_uncertainty(np.array(metric_values), np.array(group_sizes))
            uncertainty = (ci_bounds[1] - ci_bounds[0]) / 2
        else:
            metric = metric_func(self.y_pred, self.y_true)
            uncertainty = None
            metric_dict = None

        return metric, uncertainty, metric_dict

    def pearson(self):
        """
        Calculate the Pearson correlation coefficient (PCC) between the predicted and true values.
        """
        return self._metric_by_group(stats.pearsonr)

    def spearman(self):
        """
        Calculate the Spearman rank correlation coefficient between the predicted and true values.
        """
        return self._metric_by_group(stats.spearmanr)

    def kendall(self):
        """
        Calculate the Kendall's tau correlation coefficient between the predicted and true values.
        """
        return self._metric_by_group(stats.kendalltau)

    def mse(self):
        """
        Calculate the mean square error (MSE) between the predicted and true values.
        """
        return self._metric_by_group(lambda y_pred, y_true: (np.mean((y_pred - y_true) ** 2), None))

    def rmse(self):
        """
        Calculate the root mean square error (RMSE) between the predicted and true values.
        """
        return self._metric_by_group(lambda y_pred, y_true: (np.sqrt(np.mean((y_pred - y_true) ** 2)), None))

    def c_index(self):
        """
        Calculate the concordance index (C-index) between the predicted and true values.
        """
        return self._metric_by_group(lambda y_pred, y_true: (calc_c_index(y_pred, y_true), None))

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
    
    c_index = S / z if z > 0 else 0.0

    return c_index

