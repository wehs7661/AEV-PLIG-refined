import os
import sys
import time
import torch
import random
import pickle
import datetime
import argparse
import aev_plig
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from aev_plig.model import GATv2Net
from aev_plig.utils import utils, nn_utils, calc_metrics


def initialize(args):
    parser = argparse.ArgumentParser(
        description="This CLI trains an AEV-PLIG model on a dataset."
    )
    parser.add_argument(
        '-p',
        '--prefix',
        type=str,
        default='dataset',
        help='The prefix of the dataset name.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=128,
        help='The batch size for training. The default is 128.'
    )
    parser.add_argument(
        '-n',
        '--n_epochs',
        type=int,
        default=200,
        help='The number of epochs for training. The default is 200.'
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
        '-lr',
        '--learning_rate',
        type=float,
        default=0.00012291937615434127,
        help='The learning rate. The default is 0.00012291937615434127.'
    )
    parser.add_argument(
        '-a',
        '--act_fn',
        type=str,
        default='leaky_relu',
        help='The activation function. The default is leaky_relu.'
    )
    parser.add_argument(
        '-nm',
        '--n_models',
        type=int,
        default=5,
        help='The number of models to train. The default is 5.'
    )
    parser.add_argument(
        '-s',
        '--seeds',
        type=int,
        nargs='+',
        help='The random seed(s) to use for training. By default, list(range(1, n_models+1)) is used.\
            where n_models is the number of models to train, as specified by the -nm/--n_models flag.'
    )
    parser.add_argument(
        '-l',
        '--log',
        type=str,
        default='train_aev_plig.log',
        help='The path to the log file. The default is train_aev_plig.log.'
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='outputs',
        help='The directory to save the trained models and analysis files. The default is outputs.'
    )

    args = parser.parse_args(args)
    return args

def predict(model, device, loader, y_scaler=None):
    """
    Make predictions on the given dataset

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction.
    device : torch.device
        The device to use for prediction, e.g. 'cpu' or 'cuda'.
    loader : torch.utils.data.DataLoader
        The data loader for the dataset.
    y_scaler : sklearn.preprocessing.StandardScaler
        The scaler to inverse-transform predictions.

    Returns
    -------
    y_true : numpy.ndarray
        The ground truth labels.
    y_pred : numpy.ndarray
        The predicted labels.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            # Iterate over batches of data from loader
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    y_true = y_scaler.inverse_transform(total_labels.numpy().flatten().reshape(-1,1)).flatten()
    y_pred = y_scaler.inverse_transform(total_preds.detach().numpy().flatten().reshape(-1,1)).flatten()
    return y_true, y_pred


def train_one_epoch(model, device, train_loader, optimizer, loss_fn):
    """
    Trains the model for one epoch on the training dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    device : torch.device
        The device to use for training, e.g. 'cpu' or 'cuda'.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training dataset.
    optimizer : torch.optim.Optimizer
        The optimizer to use for updating model parameters
    loss_fn : torch.nn.Module
        The loss function to use for training.

    Returns
    -------
    loss : float
        The average loss over the training dataset.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, data in tqdm(enumerate(train_loader), desc="Iterating", unit=" batch", total=len(train_loader), file=sys.__stderr__, leave=False, position=1):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += (loss.item()*len(data.y))

    loss = total_loss / len(train_loader.dataset)

    return loss


def _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, y_scaler, model_path):
    """
    Trains the model over multiple epochs, validates, and saves the best model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    device : torch.device
        The device to use for training, e.g. 'cpu' or 'cuda'.
    loss_fn : torch.nn.Module
        The loss function to use for training.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training dataset.
    valid_loader : torch.utils.data.DataLoader
        The data loader for the validation dataset.
    optimizer : torch.optim.Optimizer
        The optimizer to use for updating model parameters.
    n_epochs : int
        The number of epochs to train for.
    y_scaler : sklearn.preprocessing.StandardScaler
        The scaler to inverse-transform predictions.
    model_path : str
        The path to save the trained model.
    """
    best_pc = -1.1
    pcs = []
    for epoch in tqdm(range(n_epochs), desc="\nTraining", unit="epoch", total=n_epochs, file=sys.__stderr__, position=0):
    
        loss = train_one_epoch(model, device, train_loader, optimizer, loss_fn)
        
        y_true, y_label = predict(model, device, valid_loader, y_scaler)

        current_pc = calc_metrics.calc_pearson(y_true, y_label)
        pcs.append(current_pc)
        
        low = np.maximum(epoch-7,0)
        avg_pc = np.mean(pcs[low:epoch+1])
        if(avg_pc > best_pc):
            torch.save(model.state_dict(), model_path)
            best_pc = avg_pc  

        print(f'\n  ✅ Completed epoch {epoch + 1}/{n_epochs}')
        print(f'    - Validation loss (MSE): {loss:.7f}')
        print(f'    - Pearson correlation coefficient: {current_pc:.7f}')
        print(f"    - Kendall's tau correlation coefficient: {calc_metrics.calc_kendall(y_true, y_label):.7f}")
        print(f'    - Spearman correlation coefficient: {calc_metrics.calc_spearman(y_true, y_label):.7f}')
        print(f'    - C-index: {calc_metrics.calc_c_index(y_true, y_label):.7f}')

def train_ensemble(batch_size, learning_rate, n_epochs, prefix, hidden_dim, n_heads, act_fn, seeds, output_dir):
    """
    Trains a GATv2Net model on graph data with multiple seeds and evaluates the ensemble.

    Parameters
    ----------
    batch_size : int
        Number of graphs per batch.
    learning_rate : float
        Learning rate for the optimizer.
    n_epochs : int
        Number of training epochs.
    prefix : str
        Dataset prefix (e.g., 'dataset').
    hidden_dim : int
        Hidden layer size.
    n_heads : int
        Number of attention heads.
    act_fn : str
        Activation function name.
    seeds : list of int
        List of random seeds for training multiple models.
    output_dir : str
        Directory to save the trained models.
    """
    device = nn_utils.get_device()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_data = nn_utils.GraphDataset(root='data', dataset=f'{prefix}_train', y_scaler=None)
    valid_data = nn_utils.GraphDataset(root='data', dataset=f'{prefix}_validation', y_scaler=train_data.y_scaler)
    test_data = nn_utils.GraphDataset(root='data', dataset=f'{prefix}_test', y_scaler=train_data.y_scaler)
    print(f"Number of node features: {train_data.num_node_features}")
    print(f"The number of edge features: {train_data.num_edge_features}")

    for i, seed in enumerate(seeds):
        print(f'\n🏋️ Training model {i + 1}/{len(seeds)} with seed {seed} ...')
        nn_utils.set_seed(seed)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        config = argparse.Namespace(
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            prefix=prefix,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            act_fn=act_fn
        )

        model = GATv2Net(
            node_feature_dim=train_data.num_node_features,
            edge_feature_dim=train_data.num_edge_features,
            config=config
        )
        model.apply(nn_utils.init_weights)
    
        weight_decay = 0
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
        model_file_name = f'{timestr}_model_{GATv2Net.__name__}_{prefix}_{i}.model'
        model_path = os.path.join(output_dir, model_file_name)
        model.to(device)
        _train(
            model=model,
            device=device,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            n_epochs=n_epochs,
            y_scaler=train_data.y_scaler,
            model_path=model_path
        )
        
        print(f'\nFinished training model {i + 1}! 🍺🍺🍺')
        model.load_state_dict(torch.load(model_path))
        y_true, y_pred = predict(model, device, test_loader, train_data.y_scaler)

        if i == 0:
            df_test = pd.DataFrame(data=y_true, index=range(len(y_true)), columns=['truth'])
        
        col = 'preds_' + str(i)
        df_test[col] = y_pred

        print(f"  - Test RMSE: {calc_metrics.calc_rmse(y_true, y_pred):.7f}")
        print(f"  - Test Pearson correlation: {calc_metrics.calc_pearson(y_true, y_pred):.7f}")
        print(f"  - Test Kendall's tau correlation: {calc_metrics.calc_kendall(y_true, y_pred):.7f}")
        print(f"  - Test Spearman correlation: {calc_metrics.calc_spearman(y_true, y_pred):.7f}")
        print(f"  - Test C-index: {calc_metrics.calc_c_index(y_true, y_pred):.7f}")
    
    df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)

    scaler_file = os.path.join(output_dir, f'{timestr}_model_{GATv2Net.__name__}_{prefix}.pickle')   
    with open(scaler_file,'wb') as f:
        pickle.dump(train_data.y_scaler, f)
    
    test_preds = np.array(df_test['preds'])
    test_truth = np.array(df_test['truth'])
    test_ens_pc = calc_metrics.calc_pearson(test_truth, test_preds)
    test_ens_rmse = calc_metrics.calc_rmse(test_truth, test_preds)
    section_str = "\nTest results for the ensemble model"
    print(section_str + "\n" + "=" * (len(section_str) - 1))
    print(f"RMSE: {test_ens_rmse:.7f}")
    print(f"Pearson correlation: {test_ens_pc:.7f}")
    print(f"Kendall's tau correlation: {calc_metrics.calc_kendall(test_truth, test_preds):.7f}")
    print(f"Spearman correlation: {calc_metrics.calc_spearman(test_truth, test_preds):.7f}")
    print(f"C-index: {calc_metrics.calc_c_index(test_truth, test_preds):.7f}")

def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    if args.seeds is None:
        args.seeds = list(range(1, args.n_models + 1))
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}")

    train_ensemble(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        prefix=args.prefix,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        act_fn=args.act_fn,
        seeds=args.seeds,
        output_dir=args.output_dir
    )
    print(f"\nElapsed time: {utils.format_time(time.time() - t0)}")
