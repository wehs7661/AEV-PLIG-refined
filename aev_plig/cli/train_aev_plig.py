import os
import sys
import time
import torch
import random
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch_geometric.loader import DataLoader
from aev_plig.helpers import rmse, pearson, model_dict
from aev_plig.utils import GraphDataset, init_weights


def initialize(args):
    parser = argparse.ArgumentParser(
        description="This CLI trains an AEV-PLIG model on a dataset."
    )
    parser.add_argument('--model', type=str, default='GATv2Net')
    parser.add_argument('--dataset', type=str, default='pdbbind_U_bindingnet_ligsim90')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--head', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00012291937615434127)
    parser.add_argument('--activation_function', type=str, default='leaky_relu')
    args = parser.parse_args(args)
    return args

def predict(model, device, loader, y_scaler=None):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return y_scaler.inverse_transform(total_labels.numpy().flatten().reshape(-1,1)).flatten(), y_scaler.inverse_transform(total_preds.detach().numpy().flatten().reshape(-1,1)).flatten()


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    log_interval = 100
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
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]'.format(epoch,
                                                             batch_idx * len(data.y),
                                                             len(train_loader.dataset),
                                                             100. * batch_idx / len(train_loader)))
    
    print("Loss for epoch {}: {:.4f}".format(epoch, total_loss/len(train_loader.dataset)))
    return total_loss/len(train_loader.dataset)


def _train(model, device, loss_fn, train_loader, valid_loader, optimizer, n_epochs, y_scaler, model_output_dir, model_file_name):
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
            
        print('The current validation set Pearson correlation:', current_pc)
    return


def train_NN(args):
    modeling = model_dict[args.model]
    model_st = modeling.__name__
    
    batch_size = args.batch_size
    LR = args.lr
    n_epochs = args.epochs

    print('Train for {} epochs: '.format(n_epochs))

    dataset = args.dataset

    print('Running dataset {} on model {}.'.format(dataset, model_st))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join("output", "trained_models")
    
    train_data = GraphDataset(root='data', dataset=dataset+'_train', y_scaler=None)
    valid_data = GraphDataset(root='data', dataset=dataset+'_valid', y_scaler=train_data.y_scaler)
    test_data = GraphDataset(root='data', dataset=dataset+'_test', y_scaler=train_data.y_scaler)

    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
    for i,seed in enumerate(seeds):
        random.seed(seed)
        torch.manual_seed(int(seed))
        
        model_file_name = timestr + '_model_' + model_st + '_' + dataset + '_' + str(i) + '.model'

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        if(torch.cuda.is_available()):
            print("GPU is available")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print('Device state:', device)

        model = modeling(node_feature_dim=train_data.num_node_features, edge_feature_dim=train_data.num_edge_features, config=args)
        model.apply(init_weights)
    
        print("The number of node features is ", train_data.num_node_features)
        print("The number of edge features is ", train_data.num_edge_features)
    
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

    scaler_file = 'output/trained_models/' + timestr + '_model_' + model_st + '_' + dataset + '.pickle'
    with open(scaler_file,'wb') as f:
        pickle.dump(train_data.y_scaler, f)
    
    test_preds = np.array(df_test['preds'])
    test_truth = np.array(df_test['truth'])
    test_ens_pc = pearson(test_truth, test_preds)
    test_ens_rmse = rmse(test_truth, test_preds)
    print("Ensemble test PC:", test_ens_pc)
    print("Ensemble test RMSE:", test_ens_rmse)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    t0 = time.time()
    args = initialize(sys.argv[1:])
    # sys.stdout = utils.Logger(args.log)
    # sys.stderr = utils.Logger(args.log)
    train_NN(args)
    print(f"\nElapsed time: {utils.format_time(time.time() - t0)}")
