import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import BatchNorm

activation_function_dict = {"relu": F.relu,
                            "leaky_relu": F.leaky_relu}


class GATv2Net(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, config):
        super(GATv2Net, self).__init__()
        
        self.number_GNN_layers = 5
        self.act = config.activation_function
        self.activation = activation_function_dict[self.act]
        
        self.GNN_layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()
        
        input_dim = node_feature_dim
        head = config.head
        hidden_dim = config.hidden_dim
        
        self.GNN_layers.append(GATv2Conv(input_dim, hidden_dim, heads=head, edge_dim=edge_feature_dim))
        self.BN_layers.append(BatchNorm(hidden_dim * head))
        for i in range(1, self.number_GNN_layers):
            self.GNN_layers.append(GATv2Conv(hidden_dim * head, hidden_dim, heads=head, edge_dim=edge_feature_dim))
            self.BN_layers.append(BatchNorm(hidden_dim * head))
            
        final_dim = hidden_dim * head
                
        # MLPs
        self.fc1 = nn.Linear(final_dim * 2, 1024)
        self.bn_connect1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_connect2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn_connect3 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 1)



    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for i, (layer, bn) in enumerate(zip(self.GNN_layers, self.BN_layers)):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)
        
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        # add some dense layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn_connect1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn_connect2(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.bn_connect3(x)
        return self.out(x)


    
