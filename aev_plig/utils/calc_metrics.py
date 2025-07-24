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
    def __init__(self, y_pred, y_true, groups=None, n_min=10, n_iterations=500, seed=None):
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
        n_iterations : int, optional
            The number of bootstrap iterations to perform. Default is 500.
        seed : int, optional
            The random seed for reproducibility of uncertainty estimation with bootstrapping. If None, no seed is set.
        """
        if seed is not None:
            np.random.seed(seed)
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)
        self.groups = np.array(groups) if groups is not None else None
        self.n_min = n_min
        self.n_iterations = n_iterations

        # Cache for bootstrap results to avoid redundant calculations
        self._bootstrap_cache = None
        self._group_metrics_cache = None

        if len(self.y_pred) != len(self.y_true):
            raise ValueError("y_pred and y_true must have the same length.")
        
        if self.groups is not None and len(self.y_pred) != len(self.groups):
            raise ValueError("y_pred and groups must have the same length.")

    @staticmethod
    def calc_weighted_mean(values, weights):
        """
        Calculates the weighted mean of a set of values.
        
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

    def _calc_bootstrap_uncertainty(self):
        """
        Calculates the uncertainty of metrics using bootstrapping. If groups are provided, each group is
        bootstrapped independently and uncertainties are calculated for group-weighted metrics. Otherwise,
        bootstrap sampling is done across the entire dataset.

        Returns
        -------
        bootstrap_results : dict
            Dictionary with bootstrap statistics for all metrics containing mean, std, and 95% confidence intervals.
        """
        metrics_list = ['pearson', 'spearman', 'kendall', 'mse', 'rmse', 'c_index']
        bootstrap_metrics = {metric: np.zeros(self.n_iterations) for metric in metrics_list}

        if self.groups is not None:
            # Filter out groups with less than n_min samples
            unique_groups, group_counts = np.unique(self.groups, return_counts=True)
            valid_groups = unique_groups[group_counts >= self.n_min]
            
            if len(valid_groups) == 0:
                print("No groups meet the minimum size requirement.")
                return {metric: {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan} 
                        for metric in metrics_list}
            
            # Pre-compute group indices for efficiency
            group_indices = {group: np.where(self.groups == group)[0] for group in valid_groups}
            group_sizes = {group: len(indices) for group, indices in group_indices.items()}
            
            for i in tqdm(range(self.n_iterations), total=self.n_iterations, desc="Bootstrapping metrics", file=sys.__stderr__):
                # Initialize metrics and weights for current iteration
                family_metrics = {metric: [] for metric in metrics_list}
                family_weights = []
                
                # For each valid group, sample n_i complexes with replacement from that group
                for group in valid_groups:
                    n_i = group_sizes[group]
                    original_indices = group_indices[group]
                    
                    # Sample n_i complexes with replacement from this group
                    bootstrap_indices = np.random.choice(original_indices, size=n_i, replace=True)
                    group_y_pred = self.y_pred[bootstrap_indices]
                    group_y_true = self.y_true[bootstrap_indices]
                    
                    # Calculate all metrics for this group
                    mse_val = np.mean((group_y_pred - group_y_true) ** 2)
                    family_metrics['pearson'].append(stats.pearsonr(group_y_pred, group_y_true)[0])
                    family_metrics['spearman'].append(stats.spearmanr(group_y_pred, group_y_true)[0])
                    family_metrics['kendall'].append(stats.kendalltau(group_y_pred, group_y_true)[0])
                    family_metrics['mse'].append(mse_val)
                    family_metrics['rmse'].append(np.sqrt(mse_val))
                    family_metrics['c_index'].append(calc_c_index(group_y_pred, group_y_true))
                    family_weights.append(n_i)
                
                # Calculate weighted metrics for this bootstrap iteration
                family_weights = np.array(family_weights)
                for metric_name in bootstrap_metrics.keys():
                    metric_values = np.array(family_metrics[metric_name])
                    weighted_metric = self.calc_weighted_mean(metric_values, family_weights)
                    bootstrap_metrics[metric_name][i] = weighted_metric
        else:
            n_samples = len(self.y_pred)
            for i in tqdm(range(self.n_iterations), total=self.n_iterations, desc="Bootstrapping metrics", file=sys.__stderr__):
                # Sample data points with replacement
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_y_pred = self.y_pred[bootstrap_indices]
                bootstrap_y_true = self.y_true[bootstrap_indices]
                
                # Calculate metrics for this bootstrap iteration
                mse_val = np.mean((bootstrap_y_pred - bootstrap_y_true) ** 2)
                bootstrap_metrics['pearson'][i] = stats.pearsonr(bootstrap_y_pred, bootstrap_y_true)[0]
                bootstrap_metrics['spearman'][i] = stats.spearmanr(bootstrap_y_pred, bootstrap_y_true)[0]
                bootstrap_metrics['kendall'][i] = stats.kendalltau(bootstrap_y_pred, bootstrap_y_true)[0]
                bootstrap_metrics['mse'][i] = mse_val
                bootstrap_metrics['rmse'][i] = np.sqrt(mse_val)
                bootstrap_metrics['c_index'][i] = calc_c_index(bootstrap_y_pred, bootstrap_y_true)
        
        # Aggregate results
        results = {}
        for metric_name, values in bootstrap_metrics.items():
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                results[metric_name] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'ci_lower': np.percentile(valid_values, 2.5),
                    'ci_upper': np.percentile(valid_values, 97.5)
                }
            else:
                results[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan
                }
        
        return results

    def _get_bootstrap_results(self):
        """
        Get bootstrap results, calculating them only once and caching for future use.
        """
        if self._bootstrap_cache is None:
            self._bootstrap_cache = self._calc_bootstrap_uncertainty()
        return self._bootstrap_cache

    def _metric_by_group(self, metric_func, metric_name=None):
        """
        Calculates the metric of interest (e.g., Pearson, Spearman, Kendall, MSE, and RMSE) for each group.
        
        Parameters
        ----------
        metric_func : callable
            The correlation function to apply (e.g., stats.pearsonr, stats.spearmanr, stats.kendalltau).
            Note that the function should return a tuple where the first element is the correlation value.
        metric_name : str, optional
            The name of the metric for bootstrap lookup ('pearson', 'spearman', 'kendall', 'mse', 'rmse', 'c_index').

        Returns
        -------
        metric : float
            The calculated correlation. If groups are provided, it will be the weighted average of the correlations for each group.
        uncertainty : float
            The uncertainty of the weighted average correlation, estimated by bootstrapping.
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

        else:
            metric = metric_func(self.y_pred, self.y_true)[0]
            metric_dict = None
        
        # Get uncertainty from bootstrap results
        bootstrap_results = self._get_bootstrap_results()
        ci_lower = bootstrap_results[metric_name]['ci_lower']
        ci_upper = bootstrap_results[metric_name]['ci_upper']
        uncertainty = (ci_upper - ci_lower) / 2

        return metric, uncertainty, metric_dict

    def pearson(self):
        """
        Calculates the Pearson correlation coefficient (PCC) between the predicted and true values.
        """
        return self._metric_by_group(stats.pearsonr, 'pearson')

    def spearman(self):
        """
        Calculates the Spearman rank correlation coefficient between the predicted and true values.
        """
        return self._metric_by_group(stats.spearmanr, 'spearman')

    def kendall(self):
        """
        Calculates the Kendall's tau correlation coefficient between the predicted and true values.
        """
        return self._metric_by_group(stats.kendalltau, 'kendall')

    def mse(self):
        """
        Calculates the mean square error (MSE) between the predicted and true values.
        """
        return self._metric_by_group(lambda y_pred, y_true: (np.mean((y_pred - y_true) ** 2), None), 'mse')

    def rmse(self):
        """
        Calculates the root mean square error (RMSE) between the predicted and true values.
        """
        return self._metric_by_group(lambda y_pred, y_true: (np.sqrt(np.mean((y_pred - y_true) ** 2)), None), 'rmse')

    def c_index(self):
        """
        Calculates the concordance index (C-index) between the predicted and true values.
        """
        return self._metric_by_group(lambda y_pred, y_true: (calc_c_index(y_pred, y_true), None), 'c_index')

    def all_metrics(self):
        """
        Calculates all metrics (Pearson, Spearman, Kendall, MSE, RMSE, C-index) between the predicted and true values.
        
        Returns
        -------
        metrics : dict
            A dictionary containing all calculated metrics.
        """
        metrics = {
            'pearson': self.pearson(),
            'spearman': self.spearman(),
            'kendall': self.kendall(),
            'mse': self.mse(),
            'rmse': self.rmse(),
            'c_index': self.c_index()
        }
        return metrics

@njit
def calc_c_index(y_pred, y_true):
    """
    Calculates the concordance index (C-index) between two sets of values.
    
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
