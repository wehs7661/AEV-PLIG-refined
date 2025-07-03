import os
import sys
import time
import glob
import torch
import pickle
import natsort
import argparse
import pandas as pd
import datetime
import aev_plig
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from aev_plig.utils import nn_utils, calc_metrics
from aev_plig.model import GATv2Net
from torch_geometric.loader import DataLoader
from aev_plig.cli.train_aev_plig import predict
from aev_plig.utils import utils


def initialize(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess trained AEV-PLIG models on benchmark test datasets",
    )
    parser.add_argument(
        "-md",
        "--model_dir",
        default='outputs',
        help="Directory containing trained model files and the pickled scaler file shared by the models. \
            If not specified, the current working directory is used."
    )
    parser.add_argument(
        "-tr",
        "--train_dataset",
        default='data/processed/dataset_train.pt',
        help="The path to the PyTorch file of the training set for scaling the test data."
    )
    parser.add_argument(
        "-t",
        "--test_dataset",
        default='data/processed/dataset_test.pt',
        help="The path to the PyTorch file of the test set."
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
        help="The path to the log file. The default is train_aev_plig.log."
    )

    args = parser.parse_args(args)
    return args


def load_trained_model(model_path: str, scaler_path: str, num_node_features: int, 
                      num_edge_features: int, device: torch.device) -> Tuple[GATv2Net, Any]:
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
        PyTorch device to load the model on. The values should be 'cpu' or 'cuda'.
        
    Returns
    -------
    model : GATv2Net
        The loaded GATv2Net model with trained weights.
    scaler : Any
        The scaler used during training, typically a StandardScaler or similar.
    """    
    config = argparse.Namespace(
        hidden_dim=256,
        n_heads=3,
        act_fn='leaky_relu'
    )
    
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


def assess_single_dataset(model: GATv2Net, scaler: Any, device: torch.device,
                         data_root: str, testset_name: str, trainset_name: str,
                         model_path: str, output_path: str) -> None:
    """
    Assess model on a single test dataset.
    
    Parameters
    ----------
    model : GATv2Net
        Trained model
    scaler : Any
        Scaler used during training
    device : torch.device
        PyTorch device
    data_root : str
        Root directory containing datasets
    testset_name : str
        Name of test dataset
    trainset_name : str
        Name of train dataset
    model_pat : str
        Path to model file (for output naming)
    output_path : str
        output path
    """
    print(f"Assessing model on dataset: {testset_name}")

    test_data, test_loader = load_test_dataset(data_root, testset_name, trainset_name)
    
    # Run model assessment
    print(f"Running predictions on {testset_name}...")
    y_true, y_pred, group_ids = predict(model, device, test_loader, scaler)
    
    print(f"Completed predictions for {testset_name}")
    print(f"True values range: {min(y_true):.4f} to {max(y_true):.4f}")
    print(f"Predicted values range: {min(y_pred):.4f} to {max(y_pred):.4f}")
    
    # Calculate performance metrics
    metrics = calc_metrics.MetricCalculator(y_pred, y_true, group_ids)
    metrics_dict = {
        'pearson_correlation': metrics.pearson(),
        'dataset_name': testset_name,
        'num_samples': len(y_true)
    }
    
    # print('Pearson correlation:', metrics.pearson())
    print(f'Pearson correlation for {testset_name}: {metrics.pearson()}')
    
    # Save results to CSV
    save_results_to_csv(y_true, y_pred, group_ids, metrics_dict, output_path, model_path, testset_name)
    
    print(f"Results for {testset_name} saved to: {output_path}")


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}")
    
    model_paths = natsort.natsorted(glob.glob(os.path.join(args.model_dir, '*.model')))
    print(f"\nFound {len(model_paths)} model files in directory {args.model_dir}:")
    for i in range(len(model_paths)):
        print(f"  {os.path.basename(model_paths[i])}")
    
    scaler_path = glob.glob(os.path.join(args.model_dir, '*.pickle'))
    assert len(scaler_path) == 1, f"Expected exactly one scaler file in {args.model_dir} shared by the models, found {len(scaler_path)}"
    scaler_path = scaler_path[0]
    print(f"\nFound the scaler file in directory {args.model_dir}: {os.path.basename(scaler_path)}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    print(f"Loading the trained model and scaler file ...")
    model, scaler = load_trained_model(
        model_paths[0],
        scaler_path,
        test_data.num_node_features,
        test_data.num_edge_features,
        device
    )

    y_true, y_pred, group_ids = predict(model, device, test_loader, scaler)
    err_abs = [abs(t - p) for t, p in zip(y_true, y_pred)]
    err_sq = [(t - p)**2 for t, p in zip(y_true, y_pred)]
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group_id': group_ids,
        'absolute_error': err_abs,
        'squared_error': err_sq
    })
    
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

    metrics = calc_metrics.MetricCalculator(y_true, y_pred, group_ids)
    all_metrics = metrics.all_metrics()

    print(f"\nRMSE: {all_metrics['rmse'][0]:.7f}")
    print(f"Pearson correlation: {all_metrics['pearson'][0]:.7f}")
    print(f"Kendall's tau correlation: {all_metrics['kendall'][0]:.7f}")
    print(f"Spearman correlation: {all_metrics['spearman'][0]:.7f}")
    print(f"C-index: {all_metrics['c_index'][0]:.7f}")
