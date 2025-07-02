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

This module provides functionality to evaluate pre-trained models on benchmark datasets such as FEP benchmark and CASF2016, generating performance metrics and saving results to CSV files.
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
        help="Path to scaler pickle file from training, e.g. /home/foo.pickle"
    )
    
    parser.add_argument(
        '-d', '--data_root',
        required=True,
        help="Root directory containing test and training dataset .pt files"
    )
    
    # Optional arguments
    parser.add_argument(
        '-t', '--test_dataset',
        default='dataset_test',
        help="Name of test dataset (defaults to 'dataset_test')"
    )

    parser.add_argument(
        '-r', '--train_dataset',
        default='dataset_train',
        help="Name of train dataset (defaults to 'dataset_train'). Train dataset is required for scaling the test data to have mean=0, stddev=1."
    )

    parser.add_argument(
        '-o', '--output_path',
        default='./assessment_results.csv',
        help="Path for output CSV file containing results (defaults to working directory)"
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


def calculate_performance_metrics(y_true: List[float], y_pred: List[float], 
                                group_ids: List[str], logger: logging.Logger) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for model assessment.
    
    Args:
        y_true: True binding affinity values
        y_pred: Predicted binding affinity values  
        group_ids: Group identifiers for potential group-wise analysis
        logger: Logger for metric reporting
        
    Returns:
        Dictionary containing calculated metrics
        
    Raises:
        RuntimeError: If metric calculation fails
    """
    logger.info("Calculating performance metrics...")
    
    try:
        # Initialize metric calculator with prediction data
        metrics_calc = calc_metrics.MetricCalculator(y_pred, y_true, group_ids)
        
        # Calculate primary performance metrics
        metrics = {}
        
        # Correlation metrics - primary indicators of model performance
        metrics['pearson_correlation'] = metrics_calc.pearson()
        logger.info(f"Pearson correlation: {metrics['pearson_correlation']:.4f}")
        
        return metrics
        
    except Exception as e:
        raise RuntimeError(f"Metrics calculation failed: {str(e)}")


def save_results_to_csv(y_true: List[float], y_pred: List[float], group_ids: List[str],
                       metrics: Dict[str, float], output_path: str, logger: logging.Logger) -> None:
    """
    Save assessment results to a CSV file.
    
    This function creates a comprehensive CSV output containing individual predictions
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
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write results to CSV
        # Using a single file with multiple sections is cleaner than separate files
        with open(output_path, 'w') as f:
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
        
        logger.info(f"Successfully saved {len(results_df)} predictions and metrics to {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save results to CSV: {str(e)}")


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
        test_data, _ = load_test_dataset(args.data_root, args.test_dataset, None)
        
        # Load trained model and scaler
        logger.info("Loading trained model...")
        model, scaler = load_trained_model(
            args.model_path,
            args.scaler_path,
            test_data.num_node_features,
            test_data.num_edge_features,
            device
        )
        
        # Reload test dataset with proper scaler
        test_data, test_loader = load_test_dataset(args.data_root, args.test_dataset, args.train_dataset)
        
        # Run model assessment
        logger.info("Running model assessment...")
        y_true, y_pred, group_ids = predict(model, device, test_loader, scaler)
        logger.info("Completed model assessment")
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(y_true, y_pred, group_ids, logger)
        
        # Save results to CSV
        #save_results_to_csv(y_true, y_pred, group_ids, metrics, args.output_path, logger)
        
        logger.info("Model assessment completed successfully")
        
    except Exception as e:
        logger.error(f"Model assessment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()