import os
import torch
import pickle
import argparse
import pandas as pd
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from aev_plig.utils import nn_utils, calc_metrics
from aev_plig.model import GATv2Net
from torch_geometric.loader import DataLoader
from aev_plig.cli.train_aev_plig import predict

"""
Command-line interface for assessing trained AEV-PLIG models on test datasets.

This module provides functionality to evaluate pre-trained models on benchmark datasets such as
FEP benchmark and CASF2016, generating performance metrics and saving results to CSV files.
"""


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess trained AEV-PLIG models on benchmark test datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-m', '--model_path',
        required=True,
        help="Path to trained model file, e.g. /home/foo.model"
    )

    parser.add_argument(
        '-s', '--scaler_path',
        required=True,
        help="Path to scaler pickle, e.g. /home/foo.pickle"
    )
    
    parser.add_argument(
        '-d', '--data_root',
        required=True,
        help="Root directory containing test and training dataset .pt files"
    )
    
    # Optional arguments
    parser.add_argument(
        '-t', '--test_dataset',
        nargs='+',  # Allow multiple arguments
        default=['dataset_test'],
        help="Name(s) of test dataset(s). Can specify multiple datasets separated by spaces."
    )

    parser.add_argument(
        '-r', '--train_dataset',
        default='dataset_train',
        help="Name of train dataset. The train dataset is used for scaling the test data to have mean=0, stddev=1."
    )

    parser.add_argument(
        '-o', '--output_path',
        default='./',
        help="Directory to output CSV file containing results, defaults to working directory"
    )

    parser.add_argument(
        '-l', '--log_file',
        help="Path to log file (if not specified, logs only to console)"
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cpu', 'cuda'],
        help="Device to use for model inference"
    )
    
    return parser.parse_args()


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the assessment process.
    
    Args:
        log_file: Optional path to log file. If None, logs only to console.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger('assess_model')
    logger.setLevel(logging.INFO)
    
    # Create formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler - always present
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - optional
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_trained_model(model_path: str, scaler_path: str, num_node_features: int, 
                      num_edge_features: int, device: torch.device) -> Tuple[GATv2Net, Any]:
    """
    Load a pre-trained AEV-PLIG model and its associated scaler.
    
    Args:
        model_path: Path to the trained model file (.model extension)
        scaler_path: Path to the scaler pickle file 
        num_node_features: Number of node features in the graph data
        num_edge_features: Number of edge features in the graph data
        device: PyTorch device to load the model on
        
    Returns:
        Tuple of (loaded model, loaded scaler)
        
    Raises:
        FileNotFoundError: If model or scaler files don't exist
        RuntimeError: If model loading fails
    """
    # Verify files exist before attempting to load
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Create model configuration - these should ideally be stored with the model
    # For now, using default values that match the example
    config = argparse.Namespace(
        hidden_dim=256,
        n_heads=3,
        act_fn='leaky_relu'
    )
    
    # Initialize model architecture
    model = GATv2Net(
        node_feature_dim=num_node_features,
        edge_feature_dim=num_edge_features,
        config=config
    )
    
    # Load trained model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    # Load the scaler used during training
    try:
        with open(scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler from {scaler_path}: {str(e)}")
    
    return model, scaler

def load_test_dataset(data_root: str, testset_name: str, trainset_name: str) -> Tuple[nn_utils.GraphDataset, DataLoader]:
    """
    Load test dataset and create DataLoader for model evaluation.
    
    Args:
        data_root: Root directory containing the dataset files
        testset_name: Name of the test dataset
        trainset_name: Name of the train dataset

    Returns:
        Tuple of (test dataset, test data loader)
        
    Raises:
        RuntimeError: If dataset loading fails
    """
    try:
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
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset info from {data_root}: {str(e)}")


def save_results_to_csv(y_true: List[float], y_pred: List[float], group_ids: List[str],
                       metrics: Dict[str, float], output_path: str, logger: logging.Logger,
                       model_path: str, testset_name: str) -> None:
    """
    Save assessment results to a CSV file.
    
    This function creates a CSV output containing individual predictions
    and overall performance metrics. This approach is more elegant than iterative
    writing as it allows for better data organization and atomic file operations.
    
    Args:
        y_true: True binding affinity values
        y_pred: Predicted binding affinity values
        group_ids: Group/complex identifiers
        metrics: Dictionary of calculated performance metrics
        output_path: Path where CSV file should be saved
        logger: Logger for progress tracking
        
    Raises:
        RuntimeError: If CSV writing fails
    """
    logger.info(f"Saving results to CSV file: {output_path}")
    
    try:
        # Create main results DataFrame with individual predictions
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
        
        logger.info(f"Successfully saved {len(results_df)} predictions and metrics to {final_output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save results to CSV: {str(e)}")

def assess_single_dataset(model: GATv2Net, scaler: Any, device: torch.device,
                         data_root: str, testset_name: str, trainset_name: str,
                         model_path: str, output_path: str, logger: logging.Logger) -> None:
    """
    Assess model on a single test dataset.
    
    Args:
        model: Trained model
        scaler: Scaler used during training
        device: PyTorch device
        data_root: Root directory containing datasets
        testset_name: Name of test dataset
        trainset_name: Name of train dataset
        model_path: Path to model file (for output naming)
        output_path: output path
        logger: Logger instance
    """
    logger.info(f"Assessing model on dataset: {testset_name}")
    
    try:
        # Load test dataset
        test_data, test_loader = load_test_dataset(data_root, testset_name, trainset_name)
        
        # Run model assessment
        logger.info(f"Running predictions on {testset_name}...")
        y_true, y_pred, group_ids = predict(model, device, test_loader, scaler)
        
        logger.info(f"Completed predictions for {testset_name}")
        logger.info(f"True values range: {min(y_true):.4f} to {max(y_true):.4f}")
        logger.info(f"Predicted values range: {min(y_pred):.4f} to {max(y_pred):.4f}")
        
        # Calculate performance metrics
        metrics = calc_metrics.MetricCalculator(y_pred, y_true, group_ids)
        metrics_dict = {
            'pearson_correlation': metrics.pearson(),
            'dataset_name': testset_name,
            'num_samples': len(y_true)
        }
        
        # print('Pearson correlation:', metrics.pearson())
        logger.info(f'Pearson correlation for {testset_name}: {metrics.pearson()}')
        
        # Save results to CSV
        save_results_to_csv(y_true, y_pred, group_ids, metrics_dict, output_path, logger, model_path, testset_name)
        
        logger.info(f"Results for {testset_name} saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to assess dataset {testset_name}: {str(e)}")
        raise


def main():
    """
    Main function orchestrating the model assessment workflow.
    
    This function coordinates all the steps needed to assess a trained model:
    1. Parse command line arguments
    2. Set up logging
    3. Load trained model and test data
    4. Run model assessment 
    5. Calculate performance metrics
    6. Save results to CSV
    """
    # Parse command line arguments
    args = parse_command_line_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_file)
    logger.info("Starting AEV-PLIG model assessment")
    
    # Convert single dataset to list
    test_datasets = args.test_dataset if isinstance(args.test_dataset, list) else [args.test_dataset]
    logger.info(f"Will assess model on {len(test_datasets)} dataset(s): {', '.join(test_datasets)}")
    
    try:
        # Set up computation device
        if args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using CUDA for inference")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for inference")
        
        # Load test dataset first to get feature dimensions
        logger.info("Loading test dataset...")
        # We need to load with a dummy scaler first to get dimensions
        test_data, _ = load_test_dataset(args.data_root, test_datasets[0], args.train_dataset)
        
        # Load trained model and scaler
        logger.info("Loading trained model...")
        model, scaler = load_trained_model(
            args.model_path,
            args.scaler_path,
            test_data.num_node_features,
            test_data.num_edge_features,
            device
        )

        # Assess model on each test dataset
        successful_assessments = 0
        for testset_name in test_datasets:
            try:
                assess_single_dataset(
                    model, scaler, device,
                    args.data_root, testset_name, args.train_dataset,
                    args.model_path, args.output_path, logger
                )
                successful_assessments += 1
            except Exception as e:
                logger.error(f"Failed to assess dataset {testset_name}: {str(e)}")
                # Continue with other datasets rather than failing completely
                continue
        
        logger.info(f"Model assessment completed successfully for {successful_assessments}/{len(test_datasets)} datasets")
        
        if successful_assessments == 0:
            raise RuntimeError("All dataset assessments failed")

        
        logger.info("Model assessment completed successfully")
        
    except Exception as e:
        logger.error(f"Model assessment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()