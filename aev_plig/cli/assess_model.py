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

    """
    parser.add_argument(
        '-d',
        '--data_root',
        required=True,
        help="Root directory containing test and training dataset .pt files"
    )
    parser.add_argument(
        '-t',
        '--test_dataset',
        nargs='+',
        default=['dataset_test'],
        help="Name(s) of test dataset(s). Can specify multiple datasets separated by spaces."
    )
    parser.add_argument(
        '-r',
        '--train_dataset',
        default='dataset_train',
        help="Name of train dataset. The train dataset is used for scaling the test data to have mean=0, stddev=1."
    )
    """
    parser.add_argument(
        '-o',
        '--output_path',
        default='./',
        help="Directory to output CSV file containing results, defaults to working directory"
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

def load_test_dataset(data_root: str, testset_name: str, trainset_name: str) -> Tuple[nn_utils.GraphDataset, DataLoader]:
    """
    Load test dataset and create DataLoader for model evaluation.
    
    Parameters
    ----------
    data_root : str
        Root directory containing the dataset files.
    testset_name : str
        Name of the test dataset.
    trainset_name : str
        Name of the train dataset.

    Returns
    -------
    test_data : nn_utils.GraphDataset
        Loaded test dataset object.
    test_loader : DataLoader
        DataLoader for the test dataset, used for batching during evaluation.
    """
    train_data = nn_utils.GraphDataset(root=data_root, dataset=trainset_name, y_scaler=None)

    test_data = nn_utils.GraphDataset(
        root=data_root,
        dataset=testset_name,
        y_scaler=train_data.y_scaler
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False
    )
    
    return test_data, test_loader


def save_results_to_csv(y_true: List[float], y_pred: List[float], group_ids: List[str],
                       metrics: Dict[str, float], output_path: str,
                       model_path: str, testset_name: str) -> None:
    """
    Save assessment results to a CSV file.
    
    This function creates a CSV output containing individual predictions
    and overall performance metrics. This approach is more elegant than iterative
    writing as it allows for better data organization and atomic file operations.
    
    Parameters
    ----------
    y_true : list
        True binding affinity values.
    y_pred :
        Predicted binding affinity values.
    group_ids :
        Group identifiers for the complexes, e.g., protein family names.
    metrics : dict
        Dictionary containing overall performance metrics.
    output_path : str
        Path to save the output CSV file.
    """
    print(f"Saving results to CSV file: {output_path}.")

    results_df = pd.DataFrame({
        'group_id': group_ids,
        'true_value': y_true,
        'predicted_value': y_pred,
        'absolute_error': [abs(t - p) for t, p in zip(y_true, y_pred)],
        'squared_error': [(t - p)**2 for t, p in zip(y_true, y_pred)]
    })
    
    # Create summary metrics DataFrame
    metrics_df = pd.DataFrame([metrics])
    metrics_df.index = ['overall_metrics']
    
    # Generate filename based on model and testset names
    model_name = Path(model_path).stem
    output_dir = os.path.dirname(output_path)
    filename = f"MODEL_{model_name}_TESTSET_{testset_name}.csv"
    final_output_path = os.path.join(output_dir, filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write results to CSV
    with open(final_output_path, 'w') as f:
        # Write header comment
        f.write("# AEV-PLIG Model Assessment Results\n")
        f.write("# Generated by assess_model.py\n\n")
        
        # Write overall metrics section
        f.write("# Overall Performance Metrics\n")
        metrics_df.to_csv(f, index_label='metric_type')
        f.write("\n")
        
        # Write individual predictions section  
        f.write("# Individual Predictions\n")
        results_df.to_csv(f, index=False)
    
    print(f"Successfully saved {len(results_df)} predictions and metrics to {final_output_path}")


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

    metrics = calc_metrics.MetricCalculator(y_true, y_pred, group_ids)
    print('Pearson correlation:', metrics.pearson())

    """
    # Assess model on each test dataset
    successful_assessments = 0
    for testset_name in args.test_dataset:
        try:
            assess_single_dataset(
                model, scaler, device,
                args.data_root, testset_name, args.train_dataset,
                model_paths[0], args.output_path,
            )
            successful_assessments += 1
        except Exception as e:
            print(f"Failed to assess dataset {testset_name}: {str(e)}")
            # Continue with other datasets rather than failing completely
            continue
    
    print(f"Model assessment completed successfully for {successful_assessments}/{len(args.test_dataset)} datasets")
    
    if successful_assessments == 0:
        raise RuntimeError("All dataset assessments failed")
    print("Model assessment completed successfully")
    """
