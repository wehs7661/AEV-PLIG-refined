import os
import sys
import time
import glob
import torch
import pickle
import natsort
import argparse
import datetime
import aev_plig
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from aev_plig.utils import nn_utils, calc_metrics
from aev_plig.model import GATv2Net
from torch_geometric.loader import DataLoader
from aev_plig.cli.train_aev_plig import predict
from aev_plig.utils import utils
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Suppress PyTorch Geometric weights_only warning
warnings.filterwarnings("ignore", message=".*Weights only load failed.*")

def initialize(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess trained AEV-PLIG models on benchmark test datasets",
    )
    parser.add_argument(
        "-md",
        "--model_dir",
        default='outputs',
        help="Directory containing trained model files and the pickled scaler file shared by the models. \
            The default is 'outputs'."
    )
    parser.add_argument(
        "-tr",
        "--train_dataset",
        default='data/processed/dataset_train.pt',
        help="The path to the PyTorch file of the training set for scaling the test data.\
            The default is 'data/processed/dataset_train.pt'."
    )
    parser.add_argument(
        "-t",
        "--test_dataset",
        default='data/processed/dataset_test.pt',
        help="The path to the PyTorch file of the test set. The default is 'data/processed/dataset_test.pt'."
    )
    parser.add_argument(
        '-o',
        '--output_csv',
        default='assess_trained_models.csv',
        help="The path to the output CSV file where the assessment results will be saved. \
            The default is assess_trained_models.csv."
    )
    parser.add_argument(
        '-l',
        '--log',
        type=str,
        default='assess_trained_models.log',
        help="The path to the log file. The default is assessed_trained_models.log."
    )
    parser.add_argument(
        '-hd',
        '--hidden_dim',
        type=int,
        default=256,
        help='The hidden dimension size. The default is 256.'
    )
    parser.add_argument(
        '-nh',
        '--n_heads',
        type=int,
        default=3,
        help='The number of attention heads. The default is 3.'
    )
    parser.add_argument(
        '-a',
        '--act_fn',
        type=str,
        default='leaky_relu',
        help='The activation function. The default is leaky_relu.'
    )
    parser.add_argument(
        '-n',
        '--n_iterations',
        type=int,
        default=500,
        help='The number of bootstrap iterations to perform for wPCC uncertainty calculation. The default is 500.'
    )
    parser.add_argument(
        '-nm',
        '--n_min',
        type=int,
        default=10,
        help='The minimum number of samples required in a group to calculate weighted averages of metrics \
            across groups. The default is 10.'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        help='The random seed for the reproducibiility of uncertainty estimation with bootstrapping. \
            The default is None, which means no seed is set.'
    )

    args = parser.parse_args(args)
    return args


def load_trained_model(model_path: str, scaler_path: str, num_node_features: int, 
                      num_edge_features: int, device: torch.device, config: argparse.Namespace) -> Tuple[GATv2Net, Any]:
    """
    Load a pre-trained AEV-PLIG model and its associated scaler.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model file (.model extension).
    scaler_path : str  
        Path to the scaler pickle file.
    num_node_features : int
        Number of node features in the graph data
    num_edge_features : int
        Number of edge features in the graph data
    device : str
        PyTorch device to load the model on. The values should be 'xpu', 'cuda', or 'cpu'.
    config : argparse.Namespace
        Configuration parameters for the GATv2Net model, including hidden_dim, n_heads, and act_fn.
        
    Returns
    -------
    model : GATv2Net
        The loaded GATv2Net model with trained weights.
    scaler : Any
        The scaler used during training, typically a StandardScaler or similar.
    """    
    model = GATv2Net(
        node_feature_dim=num_node_features,
        edge_feature_dim=num_edge_features,
        config=config
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    with open(scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)

    return model, scaler


def make_predictions(model: GATv2Net, device: torch.device, data_loader: DataLoader, scaler: Any) -> pd.DataFrame:
    """
    Make predictions using the trained model on the provided data loader.
    
    Parameters
    ----------
    model : GATv2Net
        The trained GATv2Net model.
    device : torch.device
        The device to run the model on (CPU or GPU).
    data_loader : DataLoader
        DataLoader containing the test dataset.
    scaler : Any
        Scaler used to transform the target variable during training.
        
    Returns
    -------
    df_results : pd.DataFrame
        DataFrame containing the true values, predicted values, group IDs, absolute errors, and squared errors.
    """
    y_true, y_pred, group_ids = predict(model, device, data_loader, scaler)
    err_abs = [abs(t - p) for t, p in zip(y_true, y_pred)]
    err_sq = [(t - p)**2 for t, p in zip(y_true, y_pred)]
    df_results = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group_id': group_ids,
        'absolute_error': err_abs,
        'squared_error': err_sq
    })

    return df_results


def assess_ensemble(df_results_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Assess the ensemble of predictions from multiple models.
    
    Parameters
    ----------
    df_results_list : List[pd.DataFrame]
        List of DataFrames containing predictions from different models.
        
    Returns
    -------
    df_ensemble : pd.DataFrame
        DataFrame containing the ensemble predictions.
    """
    y_pred_ensemble = list(np.mean([df['y_pred'] for df in df_results_list], axis=0))
    y_true = df_results_list[0]['y_true']  # assumed same across all models
    group_ids = df_results_list[0]['group_id'] # assumed same across all models
    err_abs = [abs(t - p) for t, p in zip(y_true, y_pred_ensemble)]
    err_sq = [(t - p)**2 for t, p in zip(y_true, y_pred_ensemble)]
    
    df_ensemble = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred_ensemble,
        'group_id': group_ids,
        'absolute_error': err_abs,
        'squared_error': err_sq
    })
    df_ensemble.insert(0, 'model', 'ensemble')

    return df_ensemble


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    # Check if the pt files exist
    assert os.path.isfile(args.train_dataset), f"Training dataset file {args.train_dataset} does not exist."
    assert os.path.isfile(args.test_dataset), f"Test dataset file {args.test_dataset} does not exist."

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}")
    
    # Get models
    model_paths = natsort.natsorted(glob.glob(os.path.join(args.model_dir, '*.model')))
    print(f"\nFound {len(model_paths)} model files in directory {args.model_dir}:")
    for i in range(len(model_paths)):
        print(f"  {os.path.basename(model_paths[i])}")
    
    # Get scaler
    scaler_path = glob.glob(os.path.join(args.model_dir, '*.pickle'))
    assert len(scaler_path) == 1, f"Expected exactly one scaler file in {args.model_dir} shared by the models, found {len(scaler_path)}"
    scaler_path = scaler_path[0]
    print(f"\nFound the scaler file in directory {args.model_dir}: {os.path.basename(scaler_path)}\n")

    # Set up arguments for running inference
    device = torch.device('xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    train_data = nn_utils.GraphDataset(
        root=os.path.dirname(os.path.dirname(args.train_dataset)),
        dataset=os.path.splitext(os.path.basename(args.train_dataset))[0],
        y_scaler=None
    )
    test_data = nn_utils.GraphDataset(
        root=os.path.dirname(os.path.dirname(args.test_dataset)),
        dataset=os.path.splitext(os.path.basename(args.test_dataset))[0],
        y_scaler=train_data.y_scaler
    )
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    config = argparse.Namespace(
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        act_fn=args.act_fn
    )

    # Iterate over models, make predictions and calculate metrics
    df_results_list = []
    for i in range(len(model_paths)):
        print(f"\n🔍 Assessing model {i+1}/{len(model_paths)}: {os.path.basename(model_paths[i])} ...")
        print(f"   Loading the trained model and scaler file ...")
        model, scaler = load_trained_model(
            model_paths[i],
            scaler_path,
            test_data.num_node_features,
            test_data.num_edge_features,
            device,
            config
        )
        df_results = make_predictions(model, device, test_loader, scaler)
        df_results.insert(0, 'model', os.path.basename(model_paths[i]).split('.')[0])
        df_results_list.append(df_results)

        print(f"   Calculating metrics and performing bootstrapping with {args.n_iterations} iterations")
        metrics = calc_metrics.MetricCalculator(
            df_results['y_true'].tolist(),
            df_results['y_pred'].tolist(),
            None if df_results['group_id'].isnull().all() else df_results['group_id'].tolist(),
            n_min = args.n_min,
            n_iterations = args.n_iterations,
            seed = args.seed
        )
        all_metrics = metrics.all_metrics()
        
        print(f"\n   Metrics for model {i+1}:")
        print(f"  - Test RMSE: {all_metrics['rmse'][0]:.7f} ± {all_metrics['rmse'][1]:.7f}")
        print(f"  - Test Pearson correlation: {all_metrics['pearson'][0]:.7f} ± {all_metrics['pearson'][1]:.7f}")
        print(f"  - Test Kendall's tau correlation: {all_metrics['kendall'][0]:.7f} ± {all_metrics['kendall'][1]:.7f}")
        print(f"  - Test Spearman correlation: {all_metrics['spearman'][0]:.7f} ± {all_metrics['spearman'][1]:.7f}")
        print(f"  - Test C-index: {all_metrics['c_index'][0]:.7f} ± {all_metrics['c_index'][1]:.7f}")

    # Assess ensemble model and calculate metrics
    if len(model_paths) > 1:
        df_ensemble = assess_ensemble(df_results_list)
        df_results_list.append(df_ensemble)
        print(f"\n🔍 Calculating metrics and performing bootstrapping with {args.n_iterations} iterations for ensemble model ...")
        metrics = calc_metrics.MetricCalculator(
            df_ensemble['y_true'].tolist(),
            df_ensemble['y_pred'].tolist(),
            None if df_ensemble['group_id'].isnull().all() else df_ensemble['group_id'].tolist(),
            n_min=args.n_min,
            n_iterations=args.n_iterations,
            seed=args.seed
        )
        all_metrics = metrics.all_metrics()
        
        section_str = "\nTest results for the ensemble model"
        print(section_str + "\n" + "=" * (len(section_str) - 1))
        print(f"  - Test RMSE: {all_metrics['rmse'][0]:.7f} ± {all_metrics['rmse'][1]:.7f}")
        print(f"  - Test Pearson correlation: {all_metrics['pearson'][0]:.7f} ± {all_metrics['pearson'][1]:.7f}")
        print(f"  - Test Kendall's tau correlation: {all_metrics['kendall'][0]:.7f} ± {all_metrics['kendall'][1]:.7f}")
        print(f"  - Test Spearman correlation: {all_metrics['spearman'][0]:.7f} ± {all_metrics['spearman'][1]:.7f}")
        print(f"  - Test C-index: {all_metrics['c_index'][0]:.7f} ± {all_metrics['c_index'][1]:.7f}")

    # Flatten the dataframe and save to .csv file
    df = pd.concat(df_results_list, ignore_index=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to {args.output_csv}")
