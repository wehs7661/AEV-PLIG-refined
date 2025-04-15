import os
import sys
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

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
                 ids=None, y=None, graphs_dict=None, y_scaler=None):

        super(GraphDataset, self).__init__(root)
        self.dataset = dataset
        #torch.serialization.add_safe_globals([GraphDataset])
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
            print(f'Preparing {self.processed_paths[0]} ...')

        else:
            self.process(ids, y, graphs_dict)
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

    def process(self, ids, y, graphs_dict):
        assert (len(ids) == len(y)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            label = y[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.FloatTensor(np.array([label])))
            
            data_list.append(data_point)

        # print('Graph construction done. Saving to file.')
        #self.data, self.slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])
        #torch.save((self.data, self.slices), self.processed_paths[0])
        


class GraphDatasetPredict(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, graph_ids=None, graphs_dict=None):

        super(GraphDatasetPredict, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])

        else:
            self.process(ids, graph_ids, graphs_dict)
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

    def process(self, ids, graph_ids, graphs_dict):
        assert (len(ids) == len(graph_ids)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            graph_id = graph_ids[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.IntTensor(np.array([graph_id])))
            
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        self.save(data_list, self.processed_paths[0])

class Logger:
    """
    A logger class that redirects the STDOUT and STDERR to a specified output file while
    preserving the output on screen. This is useful for logging terminal output to a file
    for later analysis while still seeing the output in real-time during execution.

    Parameters
    ----------
    logfile : str
        The file path of which the standard output and standard error should be logged.

    Attributes
    ----------
    terminal : :code:`io.TextIOWrapper` object
        The original standard output object, typically :code:`sys.stdout`.
    log : :code:`io.TextIOWrapper` object
        File object used to log the output in append mode.
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        """
        Writes a message to the terminal and to the log file.

        Parameters
        ----------
        message : str
            The message to be written to STDOUT and the log file.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure the message is written immediately

    def flush(self):
        """
        This method is needed for Python 3 compatibility. This handles the flush command by doing nothing.
        Some extra behaviors may be specified here.
        """
        # self.terminal.log()
        pass

def format_time(t):
    """
    Converts time in seconds to a more readable format.

    Parameters
    ----------
    t : float
        The time in seconds.

    Returns
    -------
    t_str : str
        A string representing the time duration in a format of "X hour(s) Y minute(s) Z second(s)", adjusting the units
        as necessary based on the input duration, e.g., 1 hour(s) 0 minute(s) 0 second(s) for 3600 seconds and
        15 minute(s) 30 second(s) for 930 seconds.
    """
    hh_mm_ss = str(datetime.timedelta(seconds=t)).split(":")

    if "day" in hh_mm_ss[0]:
        # hh_mm_ss[0] will contain "day" and cannot be converted to float
        hh, mm, ss = hh_mm_ss[0], float(hh_mm_ss[1]), float(hh_mm_ss[2])
        t_str = f"{hh} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"
    else:
        hh, mm, ss = float(hh_mm_ss[0]), float(hh_mm_ss[1]), float(hh_mm_ss[2])
        if hh == 0:
            if mm == 0:
                t_str = f"{ss:.1f} second(s)"
            else:
                t_str = f"{mm:.0f} minute(s) {ss:.0f} second(s)"
        else:
            t_str = f"{hh:.0f} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"

    return t_str


def get_atom_types_from_sdf(sdf_file):
    """
    Get the atom types from an SDF file.

    Parameters
    ----------
    sdf_file : str
        Path to the SDF file.
    
    Returns
    -------
    atom_types : list
        List of atom types.
    """
    suppl = Chem.SDMolSupplier(sdf_file)
    atom_types = []
    for mol in suppl:
        if mol is not None:
            for atom in mol.GetAtoms():
                atom_types.append(atom.GetSymbol())
    
    atom_types = list(set(atom_types))
    return atom_types