import os
import sys
import time
import glob
import natsort
import argparse
import datetime
import aev_plig
import numpy as np
import pandas as pd
from tqdm import tqdm
from aev_plig import utils, calc_metrics
from abc import ABC, abstractmethod

def initialize(args):
    parser = argparse.ArgumentParser(
        description="This CLI processes an input dataset and returns a CSV file for graph generation."
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        required=True,
        choices=[
            "pdbbind",
            "hiqbind",
            "bindingdb",
            "bindingnet_v1",
            "bindingnet_v2",
            "bindingnet_v1_v2",
            "neuralbind",
            "custom"
        ],
        help="The dataset to process. Options include pdbbind, hiqbind, bindingdb, bindingnet_v1, bindingnet_v2, \
            neuralbind, and custom (user-defined dataset)."
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        nargs ="+",
        help="The root directory containing the dataset. Note that if bindingnet_v1_v2 is selected as the dataset, \
            (via the flag -ds/--dataset), two directories must be provided: the first one is for BindingNet v1\
            and the second one is for BindingNet v2."
    )
    parser.add_argument(
        "-cr",
        "--csv_ref",
        type=str,
        help="The reference CSV file containing ligand paths (in the column 'ligand_path') against which \
            the maximum Tanimoto similarity is calculated for each ligand in the processed dataset."
    )
    parser.add_argument(
        "-cf",
        "--csv_filter",
        type=str,
        help="The CSV file containing the system IDs (in the column 'system_id') to be filtered out from the\
        processed dataset."
    )
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="The split ratio for train, validation, and test sets. Default is [0.8, 0.1, 0.1]."
    )
    parser.add_argument(
        "-sc",
        "--similarity_cutoff",
        type=float,
        default=1.0,
        help="The cutoff value for maximum Tanimoto similarity. Default is 1.0. A value of x means that only ligands\
            with maximum Tanimoto similarity less than x will be considered for splitting. This option is only\
            valid when the flag '-s'/'--split' is specified."
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=None,
        help="The random seed for splitting the dataset. Default is None, in which case no random seed is used."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output CSV file. The default is processed_[dataset].csv where [dataset] is the dataset name."
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="process_dataset.log",
        help="The path to the log file. Default is process_dataset.log."
    )

    args = parser.parse_args(args)
    return args


class DatasetCollector(ABC):
    """
    Abstract base class for collecting dataset entries.
    """

    def __init__(self, base_dir):
        """
        Initializes the DatasetCollector with a base directory.

        Parameters
        ----------
        base_dir : str
            The base directory containing dataset files.
        """
        self.base_dir = base_dir
        self.data = []  # Will be a list of dictionaries for DataFrame construction
    
    @abstractmethod
    def collect(self):
        """
        Collect dataset entries and return as a DataFrame.
        """
        pass

    def _add_entry(self, system_id, pK, protein_path, ligand_path):
        """
        Adds a single entry to the data list with validation.

        Parameters
        ----------
        system_id : str
            Unique identifier for the protein-ligand system.
        pK : float
            Binding affinity value, which is defined as -log10(K), where K can be Kd, Ki, or IC50.
        protein_path : str
            Path to the protein structure file.
        ligand_path : str
            Path to the ligand structure file.
        """
        assert os.path.exists(protein_path), f"File {protein_path} does not exist."
        self.data.append({
            "system_id": system_id,
            "pK": pK,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
        })

class PDBBindCollector(DatasetCollector):
    """
    Collector for PDBBind dataset.
    """

    def collect(self):
        """
        Collect entries from the PDBBind dataset.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for PDBBind entries.
        """
        # 1. Load binding affinity data
        index_files = [
            os.path.join(self.base_dir, 'index', 'INDEX_refined_data.2020'),
            os.path.join(self.base_dir, 'index', 'INDEX_general_PL_data.2020'),
        ]
        binding_dict = self._load_pdbbind_indices(index_files)

        # 2. Collect file paths
        for subset_dir in ["refined-set", "v2020-other-PL"]:
            dirs = natsort.natsorted(glob.glob(os.path.join(self.base_dir, f'{subset_dir}/*')))
            for dir_path in tqdm(dirs, desc=f"Collecting {subset_dir} entries", file=sys.__stderr__):
                base_name = os.path.basename(dir_path)
                if base_name in ["index", "readme"]:
                    continue
                system_id = base_name
                pK = binding_dict.get(system_id)
                protein_path = os.path.abspath(os.path.join(dir_path, f'{base_name}_protein.pdb'))
                ligand_path = os.path.abspath(protein_path.replace('_protein.pdb', '_ligand.mol2'))
                self._add_entry(system_id, pK, protein_path, ligand_path)
        
        df = pd.DataFrame(self.data)
        return df

    @staticmethod
    def _load_pdbbind_indices(index_files):
        """
        Load and process PDBbind index files into a dictionary mapping system IDs to pK values.

        Parameters
        ----------
        index_files : list of str
            List of paths to the index files.

        Returns
        -------
        binding_dict : dict
            Dictionary mapping system_id to pK values.
        """
        df_list = []
        for file in index_files:
            rows = []
            with open(file, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split(None, 5)[:-2]
                    rows.append(parts)
            df = pd.DataFrame(rows, columns=["system_id", "resolution", "year", "pK"])
        df_list.append(df)
        binding_df = pd.concat(df_list, ignore_index=True)
        binding_dict = dict(zip(binding_df["system_id"], binding_df["pK"]))
        return binding_dict

class HiQBindCollector(DatasetCollector):
    """
    Collector for HiQBind dataset.
    """

    def collect(self):
        """
        Collect entries from the HiQBind dataset.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for HiQBind entries.
        """
        # 1. Load binding affinity data
        df_index = self._load_hiqbind_metadata()

        # 2. Collect file paths
        for _, row in tqdm(df_index.iterrows(), desc="Collecting HiQBind entries", file=sys.__stderr__):
            pdb_id = row["PDBID"]
            ligand_name = row["Ligand Name"]
            ligand_chain = row["Ligand Chain"]
            ligand_resnum = row["Ligand Residue Number"]
            pK = -row["Log Binding Affinity"]
            subset = row["subset"]

            system_id = f"{pdb_id}_{ligand_chain}_{ligand_resnum}"
            protein_path = os.path.join(
                self.base_dir,
                f"raw_data_hiq_{subset}",
                pdb_id,
                f"{pdb_id}_{ligand_name}_{ligand_chain}_{ligand_resnum}",
                f"{pdb_id}_{ligand_name}_{ligand_chain}_{ligand_resnum}_protein_refined.pdb"
            )
            ligand_path = protein_path.replace("_protein_refined.pdb", "_ligand_refined.sdf")
            self._add_entry(system_id, pK, protein_path, ligand_path)
        
        df = pd.DataFrame(self.data)
        return df

    def _load_hiqbind_metadata(self):
        """
        Load and combine HiQBind metadata from CSV files

        Returns
        -------
        df_index : pandas.DataFrame
            Combined DataFrame with HiQBind metadata and subset labels.
        """
        metadata_files = {
            "sm": "hiqbind_sm_metadata.csv",
            "poly": "hiqbind_poly_metadata.csv"
        }
        df_list = []
        for subset, filename in metadata_files.items():
            df = pd.read_csv(os.path.join(self.base_dir, filename))
            df['subset'] = subset
            df_list.append(df)
        df_index = pd.concat(df_list, ignore_index=True)
        return df_index

class BindingDBCollector(DatasetCollector):
    """
    Collector for BindingDB dataset.
    """

    def collect(self):
        """
        Collect entries from the BindingDB dataset.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for HiQBind entries.
        """
        target_dirs = [d for d in natsort.natsorted(glob.glob(os.path.join(self.base_dir, "*"))) if os.path.isdir(d)]
        for target_dir in tqdm(target_dirs, desc="Collecting BindingDB entries", file=sys.__stderr__):
            pdb_id = os.path.basename(target_dir).split('_')[0]
            csv_file = os.path.join(target_dir, f'{pdb_id}.csv')
            if not os.path.exists(csv_file):  # Some entries in BindingDB do not have a .csv file
                continue
                
            # 1. Load binding affinity data
            df = pd.read_csv(csv_file)
            df['suffix'] = df['Compound'].str.split('=').str[-1]
            df['system_id'] = os.path.basename(target_dir) + '_' + df['suffix']
            df['pK'] = df.apply(self._calculate_pK, axis=1)
            df = df.dropna(subset=['pK'])
            binding_dict = dict(zip(df['system_id'], df['pK']))
            
            # 2. Collect file paths
            protein_path = os.path.abspath(os.path.join(target_dir, f'{pdb_id}.pdb'))
            for suffix in df['suffix'].tolist():
                system_id = f"{os.path.basename(target_dir)}_{suffix}"
                ligand_path = os.path.join(target_dir, f'{pdb_id}-results_{suffix}.mol2')
                pK = binding_dict.get(system_id)
                self._add_entry(system_id, pK, protein_path, ligand_path)
        
        df = pd.DataFrame(self.data)
        return df

    @staticmethod
    def _calculate_pK(row):
        """
        Calculates the pK value from available binding affinity measurements in a BindingDB row. The priority
        order is: Kd, Ki, and then IC50. The units are assumed to be in nM.

        Parameters
        ----------
        row : pandas.Series
            A row from the BindingDB DataFrame with affinity data.

        Returns
        -------
        pK : float
            The calculated pK value. If multiple measurements are present, the median is used.
            If no valid measurements are found, None will be returned.
        """
        priority = ["Kd(nM)", "Ki(nM)", "IC50(nM)"]
        for measurement in priority:
            if pd.notna(row[measurement]):
                if isinstance(row[measurement], str):
                    if '>' in row[measurement] or '<' in row[measurement]:
                        continue
                    data = row[measurement].split(";")
                    K_median = np.median([float(d) for d in data if d.strip()])
                    if K_median <= 0:
                        continue
                    pK = -np.log10(K_median / 1e9)
                else:
                    if row[measurement] <= 0:
                        continue
                    pK = -np.log10(float(row[measurement]) / 1e9)
                return pK
        return None

class BindingNetV1Collector(DatasetCollector):
    """
    Collector for BindingNet v1 dataset.
    """

    def collect(self):
        """
        Collect entries from the BindingNet v1 dataset.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for BindingNet V1 entries.
        """
        # 1. Load binding affinity data
        index_file = os.path.join(self.base_dir, "from_chembl_client", "index", "For_ML", "BindingNet_Uw_final_median.csv")
        df = pd.read_csv(index_file, sep="\t")
        system_ids = df["unique_identify"].tolist()
        binding_dict = dict(zip(df["unique_identify"], df["-logAffi"]))
        
        # 2. Collect file paths
        for system_id in tqdm(system_ids, desc="Collecting BindingNet v1 entries", file=sys.__stderr__):
            target_chembl, pdb_id, ligand_chembl = system_id.split("_")
            protein_path = os.path.join(self.base_dir, "from_chembl_client", pdb_id, "rec_h_opt.pdb")
            ligand_path = os.path.join(
                self.base_dir,
                "from_chembl_client",
                pdb_id, 
                f"target_{target_chembl}",
                ligand_chembl, 
                f"{pdb_id}_{target_chembl}_{ligand_chembl}.sdf"
            )
            pK = binding_dict.get(system_id)
            self._add_entry(system_id, pK, protein_path, ligand_path)
        
        df = pd.DataFrame(self.data)
        return df

class BindingNetV2Collector(DatasetCollector):
    """
    Collector for BindingNet v2 dataset.
    """

    def collect(self):
        """
        Collect entries from the BindingNet v2 dataset.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for BindingNet V2 entries.
        """
        # 1. Load binding affinity data
        index_file = os.path.join(self.base_dir, "Index_for_BindingNetv1_and_BindingNetv2.csv")
        df = pd.read_csv(index_file)
        df = df[df['Dataset'] == 'BindingNet v2']
        df['subset'] = pd.cut(
            df['SHAFTS HybridScore'], 
            bins=[-float('inf'), 1.0, 1.2, float('inf')], 
            labels=['low', 'moderate', 'high'], 
            right=False  # Include the left edge, exclude the right edge
        )
        df['system_id'] = df['Target ChEMBLID'] + '_' + df['Molecule ChEMBLID']
        binding_dict = dict(zip(df['system_id'], df['-logAffi']))

        # 2. Collect file paths
        for _, row in tqdm(df.iterrows(), desc="Collecting BindingNet v2 entries", file=sys.__stderr__):
            system_id = row['system_id']
            pK = binding_dict.get(system_id)
            subset = row['subset']
            target_chembl, ligand_chembl = system_id.split("_")
            protein_path = os.path.join(
                self.base_dir, subset, 
                f"target_{target_chembl}",
                ligand_chembl, 
                "protein.pdb"
            )
            ligand_path = protein_path.replace("protein.pdb", "ligand.sdf")
            self._add_entry(system_id, pK, protein_path, ligand_path)
        
        df = pd.DataFrame(self.data)
        return df

class BindingNetV1V2Collector(DatasetCollector):
    """
    Collector for the union set of BindingNet v1 and v2 datasets. Entries from v2 are prioritized
    for overlapping system IDs.
    """
    def __init__(self, base_dir_v1, base_dir_v2):
        """
        Initializes the BindingNetV1V2Collector with base directories for v1 and v2 datasets.

        Parameters
        ----------
        base_dir_v1 : str
            The base directory containing the BindingNet v1 dataset files.
        base_dir_v2 : str
            The base directory containing the BindingNet v2 dataset files.
        """
        self.base_dir_v1 = base_dir_v1
        self.base_dir_v2 = base_dir_v2


    def collect(self):
        """
        Collect entries from both BindingNet v1 and v2 datasets.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path.
        """
        # 1. Collect v1 and v2 entries separately using their own collectors
        print('Collecting entries from BindingNet v1 ...')
        v1_collector = BindingNetV1Collector(self.base_dir_v1)
        df_v1 = v1_collector.collect()

        print('Collecting entries from BindingNet v2 ...')
        v2_collector = BindingNetV2Collector(self.base_dir_v2)
        df_v2 = v2_collector.collect()

        # 2. Drop overlapping entries in v1
        # Note that the system_id in v1 is in the format of "target_chemblid_pdbid_ligand_chemblid"
        # and in v2 is in the format of "target_chemblid_ligand_chemblid"
        v2_ids = set(df_v2['system_id'])
        split_cols = df_v1['system_id'].str.split("_", expand=True)
        df_v1['system_id_simplified'] = split_cols[0] + "_" + split_cols[2]
        df_v1_unique = df_v1[~df_v1['system_id_simplified'].isin(v2_ids)]
        
        df_overlap = df_v1[df_v1['system_id_simplified'].isin(v2_ids)]
        print(f'Number of entries that are only in BindingNet v1: {len(df_v1_unique)}')
        print(f'Number of entries that are only in BindingNet v2: {len(df_v2) - len(df_overlap)}')
        print(f'Number of entries that are in both BindingNet v1 and v2: {len(df_overlap)}')

        # 3. Combined v2 entries with the remaining v1 entries
        df_combined = pd.concat([df_v1_unique, df_v2], ignore_index=True)
        df_combined = df_combined.sort_values(by='system_id').reset_index(drop=True)
        df_combined = df_combined.drop(columns=['system_id_simplified'])

        return df_combined

class NeuralBindCollector(DatasetCollector):
    """
    Collector for NeuralBind dataset.
    """

    def collect(self):
        """
        Collect entries from the NeuralBind dataset. (placeholder)

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for NeuralBind entries.
        """
        # To be implemented
        pass

class CustomDatasetCollector(DatasetCollector):
    """
    Collector for a user-defined custom dataset.
    """

    def __init__(self, base_dir, config):
        """
        Initializes the CustomDatasetCollector with a base directory and configuration.

        Parameters
        ----------
        base_dir : str
            The base directory containing dataset files.
        config : dict
            Configuration dictionary with the following keys:
              - index_file: Path to the index file relative to base_dir.
              - system_id_col: Column name for system ID.
              - pK_col: Column name for pK values.
              - protein_path_template: Template string for protein file paths relative to base_dir.
              - ligand_path_template: Template string for ligand file paths relative to base_dir.
        """
        super().__init__(base_dir)
        self.index_file = os.path.join(base_dir, config["index_file"])
        self.system_id_col = config["system_id_col"]
        self.pK_col = config.get("pK_col")
        self.protein_path_template = config["protein_path_template"]
        self.ligand_path_template = config["ligand_path_template"]

    def collect(self):
        """Collect entries from a custom dataset based on the provided config.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with system_id, pK, protein_path, and ligand_path for the custom dataset entries.
        """
        df = pd.read_csv(self.index_file)
        for _, row in df.iterrows():
            system_id = row[self.system_id_col]
            pK = row[self.pK_col] if self.pK_col and pd.notna(row[self.pK_col]) else None
            protein_path = os.path.join(self.base_dir, self.protein_path_template.format(system_id=system_id, **row.to_dict()))
            ligand_path = os.path.join(self.base_dir, self.ligand_path_template.format(system_id=system_id, **row.to_dict()))
            self._add_entry(system_id, pK, protein_path, ligand_path)
        return pd.DataFrame(self.data)


def collect_entries(base_dir, dataset, config=None):
    """
    Factory function to collect entries from the specified dataset.
    
    Parameters
    ----------
    base_dir : str
        The base directory containing the dataset files.
    dataset : str
        The name of the dataset to process. Options include pdbbind, hiqbind, bindingdb, bindingnet_v1,
        bindingnet_v2, neuralbind, and custom.
    config : dict
        A configuration dictionary for custom datasets. Required keys are:
          - index_file: Path to the index file relative to base_dir.
          - system_id_col: Column name for system ID.
          - pK_col: Column name for pK values.
          - protein_path_template: Template string for protein file paths.
          - ligand_path_template: Template string for ligand file paths.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with system_id, pK, protein_path, and ligand_path for the specified dataset.
    """
    collectors = {
        "pdbbind": PDBBindCollector,
        "hiqbind": HiQBindCollector,
        "bindingdb": BindingDBCollector,
        "bindingnet_v1": BindingNetV1Collector,
        "bindingnet_v2": BindingNetV2Collector,
        "bindingnet_v1_v2": BindingNetV1V2Collector,
        "neuralbind": NeuralBindCollector,
        "custom": CustomDatasetCollector,
    }
    
    if dataset not in collectors:
        raise ValueError(f"Invalid dataset: {dataset}. Available options are: {', '.join(collectors.keys())}.")
    
    if dataset == "bindingnet_v1_v2":
        base_dir_v1, base_dir_v2 = base_dir
        collector = BindingNetV1V2Collector(base_dir_v1, base_dir_v2)
    elif dataset == "custom":
        if not config:
            raise ValueError("Custom dataset requires a configuration dictionary.")
        collector = collectors[dataset](base_dir, config)
    else:
        collector = collectors[dataset](base_dir)

    df = collector.collect()

    return df


def split_dataset(df, split_ratio, random_seed=None):
    """
    Split the given dataset into train, validation, and test splits.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to be split.
    split_ratio : tuple
        A tuple of floats specifying the proportions for train, validation, and test splits.
        For example, (0.9, 0.1, 0) means 90% for train, 10% for validation, and 0% for test.
        The input tuple must have 3 elements.
    random_seed : int
        The random seed for reproducibility. Default is None, meaning that no seed is set.

    Returns 
    -------
    df : pandas.DataFrame
        The original DataFrame with an additional "split" column indicating the split.
    """
    df = df.copy()  # avoid modifying the original DataFrame
    if 'split' not in df.columns:
        df['split'] = ''

    if random_seed is not None:
        np.random.seed(random_seed)

    if len(split_ratio) != 3:
        raise ValueError("The parameter 'split_ratio' must have 3 elements.")
    ratios = [r/sum(split_ratio) for r in split_ratio]

    unassigned_mask = df['split'] == ''
    unassigned_indices = df[unassigned_mask].index

    n_unassigned = len(unassigned_indices)
    n_train = 0 if ratios[0] == 0 else int(n_unassigned * ratios[0])
    n_val = 0 if ratios[1] == 0 else int(n_unassigned * ratios[1])
    n_test = 0 if ratios[2] == 0 else n_unassigned - n_train - n_val
    if n_unassigned != n_train + n_val + n_test:
        # Put the remaining entries into the training set (the training ratio probably would not be 0 anyway)
        n_train = n_unassigned - n_test - n_val
    assert n_train + n_val + n_test == n_unassigned

    shuffled_indices = np.random.permutation(unassigned_indices)
    df.loc[shuffled_indices[:n_train], 'split'] = 'train'
    df.loc[shuffled_indices[n_train:n_train + n_val], 'split'] = 'validation'
    df.loc[shuffled_indices[n_train + n_val:], 'split'] = 'test'

    # Assert that all rows has a value for the 'split' column
    assert df['split'].notna().all()

    return df


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    log_dir = os.path.dirname(os.path.abspath(args.log))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}\n")

    # 1. Collect entries from the dataset
    print(f"Processing {args.dataset} ...\n")
    args.output = f"processed_{args.dataset}.csv" if args.output is None else args.output    
    if args.dataset == "bindingnet_v1_v2":
        df = collect_entries(args.dir, args.dataset)
    else:
        df = collect_entries(args.dir[0], args.dataset)
    
    # 2. Filter the dataset
    dropped_df = pd.DataFrame(columns=list(df.columns) + ['reason'])

    # 2.1. Drop entries that are also present in csv_filter
    if args.csv_filter:
        print(f"Filtering out entries present in {args.csv_filter} ...")
        filter_df = pd.read_csv(args.csv_filter)
        filter_ids = filter_df["system_id"].tolist()
        to_drop = df[df["system_id"].isin(filter_ids)].copy()
        to_drop['reason'] = f'Overalp with {os.path.basename(args.csv_filter)}'
        dropped_df = pd.concat([dropped_df, to_drop])
        df = df[~df["system_id"].isin(filter_ids)]
        print(f"Dropped {len(to_drop)} entries from that were also present in dataset {os.path.basename(args.csv_filter)}.")

    # 2.2. Drop entries with rare elements in the ligand
    df['atom_types'] = utils.get_atom_types_from_sdf_parallelized(df['ligand_path'].tolist())
    allowed_elements = set(['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S'])
    mask_uncommon = df['atom_types'].apply(lambda x: not set(x).issubset(allowed_elements))
    to_drop = df[mask_uncommon].copy()
    to_drop['reason'] = 'Rare elements in ligand'
    dropped_df = pd.concat([dropped_df, to_drop])
    df = df[~mask_uncommon]
    df = df.drop(columns=['atom_types'])
    print(f"Dropped {len(to_drop)} entries with rare elements in the ligand.")

    # 2.3. Drop entries with high similarity to the reference dataset
    if args.csv_ref:
        print(f"Generating fingerprints for all ligands in {args.csv_ref} and {args.dataset} ...")
        df_ref = pd.read_csv(args.csv_ref)
        fps_ref, _ = calc_metrics.generate_fingerprints_parallelized(df_ref, "ligand_path")
        fps, valid_indices = calc_metrics.generate_fingerprints_parallelized(df, "ligand_path")
        
        print(f"\nCalculating maximum Tanimoto similarity to {os.path.basename(args.csv_ref)} for each ligand in {args.dataset} ...")
        max_sims = calc_metrics.calc_max_tanimoto_similarity(fps, fps_ref)
        df['max_tanimoto_ref'] = np.nan
        df.iloc[valid_indices, df.columns.get_loc('max_tanimoto_ref')] = max_sims
    
        if args.similarity_cutoff < 1.0:
            to_drop = df[df['max_tanimoto_ref'] > args.similarity_cutoff].copy()
            to_drop['reason'] = f'Max Tanimoto similarity > {args.similarity_cutoff}'
            dropped_df = pd.concat([dropped_df, to_drop])
            df = df[df['max_tanimoto_ref'] <= args.similarity_cutoff]
            print(f"Dropped {len(to_drop)} entries with maximum Tanimoto similarity > {args.similarity_cutoff} with respect to {os.path.basename(args.csv_ref)}.")

    # 2.4. Drop entries with nan values in any column
    to_drop = df[df.isna().any(axis=1)].copy()
    to_drop['reason'] = 'NaN values'
    dropped_df = pd.concat([dropped_df, to_drop])
    df = df.dropna()
    print(f"Dropped {len(to_drop)} entries with NaN values in any column.")

    print(f"Splitting the dataset ...")
    df = split_dataset(df, args.split, random_seed=args.random_seed)
    print('  - Number of train entries:', len(df[df['split'] == 'train']))
    print('  - Number of validation entries:', len(df[df['split'] == 'validation']))
    print('  - Number of test entries:', len(df[df['split'] == 'test']))

    df.to_csv(args.output, index=False)
    dropped_df.to_csv(f"{args.output.replace('.csv', '_dropped.csv')}", index=False)
    print(f"\nProcessed dataset saved to {args.output}")
    print(f"Dropped entries saved to {args.output.replace('.csv', '_dropped.csv')}")
    print(f"Elapsed time: {utils.format_time(time.time() - t0)}")
