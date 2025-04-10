import os
import sys
import time
import glob
import natsort
import argparse
import datetime
import numpy as np
import pandas as pd
import aev_plig
from aev_plig.utils import Logger
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
        choices=["pdbbind", "hiqbind", "bindingdb", "bindingnet_v1", "bindingnet_v2", "neuralbind"],
        help="The dataset to process. Options include pdbbind, bindingnet1, bindingnet2, bindingdb, and neuralbind."
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="The root directory containing the dataset."
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
            for dir_path in dirs:
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
        for _, row in df_index.iterrows():
            pdb_id = row["PDBID"]
            ligand_name = row["Ligand Name"]
            ligand_chain = row["Ligand Chain"]
            ligand_resnum = row["Ligand Residue Number"]
            pK = row["Log Binding Affinity"]
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
        for target_dir in target_dirs:
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
        for system_id in system_ids:
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
        for _, row in df.iterrows():
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


def collect_entries(base_dir, dataset):
    """
    Factory function to collect entries from the specified dataset.
    
    Parameters
    ----------
    base_dir : str
        The base directory containing the dataset files.
    dataset : str
        The name of the dataset to process. Options include pdbbind, hiqbind, bindingdb, bindingnet_v1,
        bindingnet_v2, and neuralbind.

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
        # "neuralbind": NeuralBindCollector
    }
    
    if dataset not in collectors:
        raise ValueError(f"Invalid dataset: {dataset}. Available options are: {', '.join(collectors.keys())}.")
    
    collector = collectors[dataset](base_dir)
    df = collector.collect()
    return df


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = Logger(args.log)
    sys.stderr = Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}\n")

    print(f"Processing {args.dataset} ...")
    args.output = f"processed_{args.dataset}.csv" if args.output is None else args.output
    df = collect_entries(args.dir, args.dataset)
    df.to_csv(args.output, index=False)
    print(f"Elapsed time: {time.time() - t0:.2f} seconds")
