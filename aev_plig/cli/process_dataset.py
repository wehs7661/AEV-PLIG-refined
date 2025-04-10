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


def calculate_pK(row):
    """
    Calculates the pK value from available binding affinity measurements. The priority order is;
    Kd, Ki, and IC50. The units are assumed to be in nM.

    Parameters
    ----------
    row : pandas.Series
        A row from the binding affinity DataFrame.
    
    Returns
    -------
    pK: float
        The calculated pK value.
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


def collect_entries(base_dir, dataset=None):
    data = []
    if dataset == "pdbbind":
        # Get the binding affinity data
        index_files = [
            os.path.join(base_dir, 'index', 'INDEX_refined_data.2020'),
            os.path.join(base_dir, 'index', 'INDEX_general_PL_data.2020'),
        ]
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
        
        # Get the file paths
        for subset_dir in ["refined-set", "v2020-other-PL"]:
            dirs = natsort.natsorted(glob.glob(os.path.join(base_dir, f'{subset_dir}/*')))
            for d in dirs:
                base_name = os.path.basename(d)
                if base_name not in ["index", "readme"]:
                    system_id = base_name
                    pK = binding_dict.get(system_id)
                    protein_path = os.path.abspath(os.path.join(d, f'{base_name}_protein.pdb'))
                    ligand_path = os.path.abspath(protein_path.replace('_protein.pdb', '_ligand.mol2'))
                    data.append({
                        "system_id": system_id,
                        "pK": pK,
                        "protein_path": protein_path,
                        "ligand_path": ligand_path,
                    })

    elif dataset == "hiqbind":
        # Get the binding affinity data
        index_sm = os.path.join(base_dir, "hiqbind_sm_metadata.csv")
        df_sm = pd.read_csv(index_sm)
        df_sm['subset'] = "sm"

        index_poly = os.path.join(base_dir, "hiqbind_poly_metadata.csv")
        df_poly = pd.read_csv(index_poly)
        df_poly['subset'] = "poly"

        df = pd.concat([df_sm, df_poly], ignore_index=True)

        # Get the file paths
        for _, row in df.iterrows():
            pdb_id = row["PDBID"]
            ligand_name = row["Ligand Name"]
            ligand_chain = row["Ligand Chain"]
            ligand_resnum = row["Ligand Residue Number"]
            pK = row["Log Binding Affinity"]
            subset = row["subset"]

            system_id = f"{pdb_id}_{ligand_chain}_{ligand_resnum}"
            protein_path = os.path.abspath(os.path.join(
                base_dir,
                f"raw_data_hiq_{subset}",
                pdb_id,
                f"{pdb_id}_{ligand_name}_{ligand_chain}_{ligand_resnum}",
                f"{pdb_id}_{ligand_name}_{ligand_chain}_{ligand_resnum}_protein_refined.pdb"
            ))
            ligand_path = protein_path.replace("_protein_refined.pdb", "_ligand_refined.sdf")
            data.append({
                "system_id": system_id,
                "pK": pK,
                "protein_path": protein_path,
                "ligand_path": ligand_path,
            })

    elif dataset == "bindingdb":
        target_dirs = [d for d in natsort.natsorted(glob.glob(os.path.join(base_dir, "*"))) if os.path.isdir(d)]
        for target_dir in target_dirs:
            pdb_id = os.path.basename(target_dir).split('_')[0]
            csv_file = os.path.join(target_dir, f'{pdb_id}.csv')
            if not os.path.exists(csv_file):
                # Some entries in BindingDB do not have a .csv file
                continue
            
            # Get the binding affinity data
            df = pd.read_csv(csv_file)
            df['suffix'] = df['Compound'].str.split('=').str[-1]
            df['system_id'] = os.path.basename(target_dir) + '_' + df['suffix']
            df['pK'] = df.apply(calculate_pK, axis=1)
            df = df.dropna(subset=['pK'])
            binding_dict = dict(zip(df['system_id'], df['pK']))
            
            # Get the file paths
            protein_path = os.path.abspath(os.path.join(target_dir, f'{pdb_id}.pdb'))
            ligand_paths = [os.path.join(target_dir, f'{pdb_id}-results_{s}.mol2') for s in df['suffix'].tolist()]
            for ligand_path in ligand_paths:
                assert os.path.exists(ligand_path), f"File {ligand_path} does not exist."
                num = os.path.basename(ligand_path).split('.mol2')[0].split('_')[-1]
                system_id = f"{os.path.basename(target_dir)}_{num}"
                pK = binding_dict.get(system_id)
                data.append({
                    "system_id": system_id,
                    "pK": pK,
                    "protein_path": protein_path,
                    "ligand_path": os.path.abspath(ligand_path)
                })
            
    elif dataset == "bindingnet_v1":
        # Get the binding affinity data
        index_file = os.path.join(base_dir, "from_chembl_client", "index", "For_ML", "BindingNet_Uw_final_median.csv")
        df = pd.read_csv(index_file, sep="\t")
        system_ids = df["unique_identify"].tolist()
        binding_dict = dict(zip(df["unique_identify"], df["-logAffi"]))

        # Get the file paths
        for system_id in system_ids:
            target_chembl = system_id.split("_")[0]
            pdb_id = system_id.split("_")[1]
            ligand_chembl = system_id.split("_")[2]
            protein_path = os.path.abspath(os.path.join(base_dir, "from_chembl_client", pdb_id, 'rec_h_opt.pdb'))
            ligand_path = os.path.abspath(os.path.join(base_dir, "from_chembl_client", pdb_id, f"target_{target_chembl}", ligand_chembl, f"{pdb_id}_{target_chembl}_{ligand_chembl}.sdf"))
            pK = binding_dict.get(system_id)
            data.append({
                "system_id": system_id,
                "pK": pK,
                "protein_path": protein_path,
                "ligand_path": ligand_path
            })

    elif dataset == "bindingnet_v2":
        # Get the binding affinity data
        index_file = os.path.join(base_dir, "Index_for_BindingNetv1_and_BindingNetv2.csv")
        df = pd.read_csv(index_file)
        df = df[df['Dataset'] == 'BindingNet v2']
        df['subset'] = pd.cut(df['SHAFTS HybridScore'], bins=[-float('inf'), 1.0, 1.2, float('inf')], labels=['low', 'moderate', 'high'])
        df['system_id'] = df['Target ChEMBLID'] + '_' + df['Molecule ChEMBLID']
        system_ids = df['system_id'].tolist()
        binding_dict = dict(zip(df['system_id'], df['-logAffi']))
        
        # Get the file paths
        for _, row in df.iterrows():
            system_id = row['system_id']
            pK = binding_dict.get(system_id)
            subset = row['subset']
            target_chembl = system_id.split("_")[0]
            ligand_chembl = system_id.split("_")[1]
            protein_path = os.path.abspath(os.path.join(base_dir, subset, f"target_{target_chembl}", f"{ligand_chembl}", "protein.pdb"))
            ligand_path = protein_path.replace("protein.pdb", "ligand.sdf")
            data.append({
                "system_id": system_id,
                "pK": pK,
                "protein_path": protein_path,
                "ligand_path": ligand_path
            })

    elif dataset == "neuralbind":
        pass
    else:
        raise ValueError("Invalid dataset.")

    return data


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
    data = collect_entries(args.dir, args.dataset)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Elapsed time: {time.time() - t0:.2f} seconds")
