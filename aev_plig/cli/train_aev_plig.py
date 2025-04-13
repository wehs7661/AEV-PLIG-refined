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
from aev_plig import utils
from aev_plig.helpers import rmse, pearson, model_dict


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
        '-l',
        '--log',
        type=str,
        default='train_aev_plig.log',
        help='The path to the log file. The default is train_aev_plig.log.'
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
    print(f'Make prediction for {len(loader.dataset)} samples...')
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


def train(model, device, train_loader, optimizer, epoch, loss_fn):
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
    epoch : int
        The current epoch number (1-indexed).
    loss_fn : torch.nn.Module
        The loss function to use for training.

    Returns
    -------
    loss : float
        The average loss over the training dataset.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += (loss.item()*len(data.y))

    loss = total_loss / len(train_loader.dataset)
    print(f"Loss for epoch {epoch}: {loss:.4f}")

    return loss


def _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, y_scaler, model_output_dir, model_file_name):
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
    model_output_dir : str
        The directory to save model checkpoints.
    model_file_name : str
        The filename for the saved model.
    """
    best_pc = -1.1
    pcs = []
    for epoch in range(n_epochs):
    
        _ = train(model, device, train_loader, optimizer, epoch + 1, loss_fn)
        
        G, P = predict(model, device, valid_loader, y_scaler)

        current_pc = pearson(G, P)
        pcs.append(current_pc)
        
        low = np.maximum(epoch-7,0)
        avg_pc = np.mean(pcs[low:epoch+1])
        if(avg_pc > best_pc):
            torch.save(model.state_dict(), os.path.join(model_output_dir, model_file_name))
            best_pc = avg_pc  

        print(f'The current validation set Pearson correlation: {current_pc}')


def train_NN(args):
    """
    Trains a GATv2Net model on graph data with multiple seeds and evaluates the ensemble.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
            - batch_size (int): Number of graphs per batch.
            - learning_rate (float): Learning rate for the optimizer.
            - n_epochs (int): Number of training epochs.
            - prefix (str): Dataset prefix (e.g., 'dataset').
            - hidden_dim (int): Hidden layer size.
            - n_heads (int): Number of attention heads.
            - act_fn (str): Activation function name.
    """
    modeling = model_dict['GATv2Net']
    model_st = modeling.__name__
    
    batch_size = args.batch_size
    LR = args.learning_rate
    n_epochs = args.n_epochs

    prefix = args.prefix
    print(f'Running model {model_st} ...')
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join("output", "trained_models")
    
    train_data = utils.GraphDataset(root='data', dataset=f'{prefix}_train', y_scaler=None)
    valid_data = utils.GraphDataset(root='data', dataset=f'{prefix}_validation', y_scaler=train_data.y_scaler)
    test_data = utils.GraphDataset(root='data', dataset=f'{prefix}_test', y_scaler=train_data.y_scaler)

    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    for i,seed in enumerate(seeds):
        random.seed(seed)
        torch.manual_seed(int(seed))
        
        model_file_name = f'{timestr}_model_{model_st}_{prefix}_{i}.model'

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        if(torch.cuda.is_available()):
            print("GPU is available")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = modeling(node_feature_dim=train_data.num_node_features, edge_feature_dim=train_data.num_edge_features, config=args)
        model.apply(utils.init_weights)
    
        print(f"  - Number of node features: {train_data.num_node_features}")
        print(f"  - The number of edge features: {train_data.num_edge_features}")
    
        weight_decay = 0
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    
        model.to(device)
        _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, train_data.y_scaler, model_output_dir, model_file_name)
        
        model.load_state_dict(torch.load(os.path.join(model_output_dir, model_file_name)))
        
        G_test, P_test = predict(model, device, test_loader, train_data.y_scaler)

        if(i == 0):
            df_test = pd.DataFrame(data=G_test, index=range(len(G_test)), columns=['truth'])
        
        col = 'preds_' + str(i)
        df_test[col] = P_test
    
    df_test['preds'] = df_test.iloc[:,1:].mean(axis=1)

    scaler_file = f'output/trained_models/{timestr}_model_{model_st}_{prefix}.pickle'    
    with open(scaler_file,'wb') as f:
        pickle.dump(train_data.y_scaler, f)
    
    test_preds = np.array(df_test['preds'])
    test_truth = np.array(df_test['truth'])
    test_ens_pc = pearson(test_truth, test_preds)
    test_ens_rmse = rmse(test_truth, test_preds)
    print("Ensemble test PC:", test_ens_pc)
    print("Ensemble test RMSE:", test_ens_rmse)


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}\n")

    train_NN(args)
    print(f"\nElapsed time: {utils.format_time(time.time() - t0)}")
