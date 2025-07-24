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
from aev_plig.data import dataset
from aev_plig.utils import utils, chem_utils, calc_metrics

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
        nargs="+",
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
        "-f",
        "--filters",
        nargs='+',
        type=str,
        help="The filters to apply to the dataset before splitting it. The filters are in the format of \
            'column_name operator value'. For example, 'max_tanimoto_schrodinger < 0.9' will filter out entries \
            with column 'max_tanimoto_schrodinger' larger than 0.9. The operators include '<', '<=', '>', \
            '>=', '==', and '!='."
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
        df = dataset.collect_entries(args.dir, args.dataset)
    else:
        df = dataset.collect_entries(args.dir[0], args.dataset)
    
    # 2. Filter the dataset
    dropped_df = pd.DataFrame(columns=list(df.columns) + ['reason'])

    # 2.1. Drop entries that are also present in csv_filter
    if args.csv_filter:
        print(f"Filtering out entries present in {args.csv_filter} ...")
        filter_df = pd.read_csv(args.csv_filter)
        filter_ids = filter_df["system_id"].tolist()
        to_drop = df[df["system_id"].isin(filter_ids)].copy()
        to_drop['reason'] = f'Overlap with {os.path.basename(args.csv_filter)}'
        dropped_df = pd.concat([dropped_df, to_drop])
        df = df[~df["system_id"].isin(filter_ids)]
        print(f"Dropped {len(to_drop)} entries from that were also present in dataset {os.path.basename(args.csv_filter)}.")

    # 2.2. Drop entries with rare elements in the ligand
    df['atom_types'] = chem_utils.get_atom_types_from_sdf_parallelized(df['ligand_path'].tolist())
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
        fps_ref, _ = chem_utils.generate_fingerprints_parallelized(df_ref, "ligand_path")
        fps, valid_indices = chem_utils.generate_fingerprints_parallelized(df, "ligand_path")
        
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

    # 2.5. Apply user-defined filters, if any
    if args.filters:
        print(f"Applying user-defined filters ...")
        for filter_str in args.filters:
            column, operator, value = utils.parse_filter(filter_str)
            original_df = df.copy()
            df = utils.apply_filter(df, column, operator, value)
            to_drop = original_df[~original_df.index.isin(df.index)].copy()
            to_drop['reason'] = f"User-defined filter: {filter_str}"
            dropped_df = pd.concat([dropped_df, to_drop])
            print(f"Dropped {len(to_drop)} entries with filter '{filter_str}'.")

    print(f"Splitting the dataset ...")
    df = dataset.split_dataset(df, args.split, random_seed=args.random_seed)
    print('  - Number of train entries:', len(df[df['split'] == 'train']))
    print('  - Number of validation entries:', len(df[df['split'] == 'validation']))
    print('  - Number of test entries:', len(df[df['split'] == 'test']))

    df.to_csv(args.output, index=False)
    dropped_df.to_csv(f"{args.output.replace('.csv', '_dropped.csv')}", index=False)
    print(f"\nProcessed dataset saved to {args.output}")
    print(f"Dropped entries saved to {args.output.replace('.csv', '_dropped.csv')}")
    print(f"Elapsed time: {utils.format_time(time.time() - t0)}")
