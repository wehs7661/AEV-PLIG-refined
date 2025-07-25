import os
import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data

def get_device():
    """
    Get the device to be used for training and inference.

    Returns
    -------
    device : torch.device
        The device to be used (CPU or GPU).
    """
    if(torch.cuda.is_available()):
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is NOT available")
        device = torch.device("cpu")
    
    return device

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to be set for random number generation.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_weights(layer):
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)

class GraphDataset(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, y=None, graphs_dict=None, y_scaler=None, group_ids=None):

        super(GraphDataset, self).__init__(root)
        self.dataset = dataset
        #torch.serialization.add_safe_globals([GraphDataset])
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
            print(f'Preparing {self.processed_paths[0]} ...')

        else:
            self.process(ids, y, graphs_dict, group_ids)
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
        
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self._data.y, (self.__len__(),1)))
        self.y_scaler = y_scaler
        self._data.y = [torch.tensor(element[0]).float() for element in self.y_scaler.transform(np.reshape(self._data.y, (self.__len__(),1)))]
        

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, y, graphs_dict, group_ids):
        assert (len(ids) == len(y)), 'Number of datapoints and labels must be the same'
        assert (len(group_ids) == len(y)), 'Number of group_ids and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            label = y[i]
            group_id = group_ids[i] if group_ids[i] is not None else None
            group_id = np.array([group_id])  # Otherwise, if group_id is None, data_point won't have group_id attribute
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(
                x=torch.Tensor(np.array(features)),
                edge_index=torch.LongTensor(np.array(edge_index)).T,
                edge_attr=torch.Tensor(np.array(edge_features)),
                y=torch.FloatTensor(np.array([label])),
                group_id=group_id
            )
            data_point.group_id = group_id

            data_list.append(data_point)

        # print('Graph construction done. Saving to file.')
        # self._data, self.slices = self.collate(data_list)
        print(f'Saving processed data to {self.processed_paths[0]} ...')
        self.save(data_list, self.processed_paths[0])
        #torch.save((self.data, self.slices), self.processed_paths[0])
        

class GraphDatasetPredict(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, graph_ids=None, graphs_dict=None, group_ids=None):

        super(GraphDatasetPredict, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])

        else:
            self.process(ids, graph_ids, graphs_dict, group_ids)
            self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, graph_ids, graphs_dict, group_ids):
        assert (len(ids) == len(graph_ids)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            graph_id = graph_ids[i]
            group_id = group_ids[i] if group_ids is not None else None
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(
                x=torch.Tensor(np.array(features)),
                edge_index=torch.LongTensor(np.array(edge_index)).T,
                edge_attr=torch.Tensor(np.array(edge_features)),
                y=torch.IntTensor(np.array([graph_id])),
                group_id=group_id
            )
            
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        self.save(data_list, self.processed_paths[0])
