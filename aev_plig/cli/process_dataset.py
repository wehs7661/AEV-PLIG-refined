import os
import sys
import time
import glob
import natsort
import argparse
import datetime
import pandas as pd
import aev_plig
from aev_plig.utils import Logger

def initialize(args):
    parser = argparse.ArgumentParser(
        description="This CLI process an input dataset and returns a CSV file that works with the CLI generate_graphs."
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        required=True,
        choices=["pdbbind", "bindingnet", "bindingdb", "neuralbind"],
        help="The dataset to process."
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="The root directory containing the dataset."
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        help="A reference dataset that contains filtered entries. The output will only contain entries that are\
            in the reference dataset, with only necessary columns for graph generation."
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


def collect_entries(base_dir, dataset=None, ref_dataset=None):
    data = []
    if dataset == "pdbbind":
        for subset_dir in ["refined-set", "v2020-other-PL"]:
            dirs = natsort.natsorted(glob.glob(os.path.join(base_dir, f'{subset_dir}/*')))
            for d in dirs:
                base_name = os.path.basename(d)
                if base_name not in ["index", "readme"]:
                    protein_path = os.path.abspath(os.path.join(d, f'{base_name}_protein.pdb'))
                    ligand_path = os.path.abspath(protein_path.replace('_protein.pdb', '_ligand.mol2'))
                    data.append({
                        "system_id": base_name.split('_')[0],
                        "protein_path": protein_path,
                        "ligand_path": ligand_path
                    })

    elif dataset == "bindingnet":
        dirs = natsort.natsorted(glob.glob(os.path.join(base_dir, "from_chembl_client/*")))
        for d in dirs:
            base_name = os.path.basename(d)  # pdb id
            if base_name not in ["index", "PDBbind_minimized"]:
                target_dirs = natsort.natsorted(glob.glob(os.path.join(d, "target_CHEMBL*")))
                for target_dir in target_dirs:
                    ligand_dirs = natsort.natsorted(glob.glob(os.path.join(target_dir, "CHEMBL*")))
                    for ligand_dir in ligand_dirs:
                        target_chembl = os.path.basename(target_dir).split('_')[-1]
                        ligand_chembl = os.path.basename(ligand_dir)
                        system_id = f"{target_chembl}_{base_name}_{ligand_chembl}"
                        protein_path = os.path.abspath(os.path.join(d, 'rec_h_opt.pdb'))
                        ligand_path = os.path.abspath(os.path.join(ligand_dir, f'{base_name}_{target_chembl}_{ligand_chembl}.sdf'))
                        data.append({
                            "system_id": system_id,
                            "protein_path": protein_path,
                            "ligand_path": ligand_path
                        })

    elif dataset == "bindingdb":
        target_dirs = [d for d in natsort.natsorted(glob.glob(os.path.join(base_dir, "*"))) if os.path.isdir(d)]
        for target_dir in target_dirs:
            pdb_id = os.path.basename(target_dir).split('_')[0]
            protein_path = os.path.abspath(os.path.join(target_dir, f'{pdb_id}.pdb'))
            ligand_paths = natsort.natsorted(glob.glob(os.path.join(target_dir, f'{pdb_id}-results_*.mol2')))
            for ligand_path in ligand_paths:
                num = os.path.basename(ligand_path).split('.mol2')[0].split('_')[-1]
                system_id = f"{os.path.basename(target_dir)}_{num}"
                data.append({
                    "system_id": system_id,
                    "protein_path": protein_path,
                    "ligand_path": os.path.abspath(ligand_path)
                })

    elif dataset == "neuralbind":
        pass
    else:
        raise ValueError("Invalid dataset.")

    # Filter entries based on the reference dataset
    if ref_dataset is not None:
        ref_df = pd.read_csv(ref_dataset)
        ref_ids = ref_df["system_id"].tolist()
        data = [entry for entry in data if entry["system_id"] in ref_ids]

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
    data = collect_entries(args.dir, args.dataset, args.ref)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Elapsed time: {time.time() - t0:.2f} seconds")
