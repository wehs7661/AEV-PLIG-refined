import os
import re
import sys
import time
import glob
import pickle
import argparse
import datetime
import aev_plig
import pandas as pd
from aev_plig import utils


def initialize(args):
    parser = argparse.ArgumentParser(
        description="This CLI process pickled graphs and create PyTorch data ready for training."
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="The directory containing the pickled graphs."
    )
    parser.add_argument(
        "-c",
        "--csv_files",
        type=str,
        nargs='+',
        required=True,
        help="The paths to the input processed CSV files, each corresponding to a pickled graph. Each CSV file \
            should at least have columns including 'system_id', 'protein_path', 'ligand_path', 'pK', and 'split'."
    )
    parser.add_argument(
        "-f",
        "--filters",
        nargs='+',
        type=str,
        help="The filters to apply to the dataset. The filters are in the format of 'column_name operator value'.\
            For example, 'max_tanimoto_schrodinger < 0.9' will filter out entries with maximum Tanimoto similarity\
            to Schrodinger dataset less than 0.9. The operators include '<', '<=', '>', '>=', '==', and '!='."
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="dataset",
        help="The prefix of the output PyTorch files. Default is 'dataset'."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output directory. Default is the same as the input directory."
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="create_pytorch_data.log",
        help="The path to the log file. Default is create_pytorch_data.log."
    )

    args = parser.parse_args(args)
    return args


def parse_filter(filter_str):
    # This regex will capture: column, operator, and value.
    pattern = r'(\w+)\s*(<=|>=|==|!=|<|>)\s*(.+)'
    match = re.match(pattern, filter_str)
    if not match:
        raise ValueError(
            f"Filter '{filter_str}' is not in the correct format. "
            f"Expected format: <column> <operator> <value>. "
            f"Example: 'max_tanimoto_schrodinger < 0.9'."
        )
    column, operator, value = match.groups()
    # Convert value to a float if possible, else keep as string
    try:
        value = float(value)
    except ValueError:
        pass
    return column, operator, value


def apply_filter(df, column, operator, value):
    if operator == '<':
        return df[df[column] < value]
    elif operator == '>':
        return df[df[column] > value]
    elif operator == '<=':
        return df[df[column] <= value]
    elif operator == '>=':
        return df[df[column] >= value]
    elif operator == '==':
        return df[df[column] == value]
    elif operator == '!=':
        return df[df[column] != value]
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}\n")

    # 1. Load the pickled graphs
    pickle_files = glob.glob(os.path.join(args.dir, "*.pkl")) + glob.glob(os.path.join(args.dir, "*.pickle"))
    print(f"Pickled graphs found in {args.dir}: {pickle_files}")
    print(f"Loading pickled graphs...")
    graphs_dict = {}
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as f:
            graphs = pickle.load(f)
        graphs_dict.update(graphs)

    # 2. Load the processed CSV files
    print('Loading the processed CSV files...')
    data_to_merge = []
    for csv_file in args.csv_files:
        print(f"Processing {csv_file}...")
        csv_data = pd.read_csv(csv_file, index_col=0)
        if args.filters:
            for filter_str in args.filters:
                column, operator, value = parse_filter(filter_str)
                csv_data = apply_filter(csv_data, column, operator, value)
        csv_data = csv_data[['system_id', 'pK', 'split']]
        data_to_merge.append(csv_data)
    data = pd.concat(data_to_merge, ignore_index=True)
    print(data[['split']].value_counts())

    # 3. Create PyTorch data
    splits = ['train', 'valid', 'test']
    for split in splits:
        if split not in data['split'].unique():
            print(f"Split '{split}' not found in the dataset. Skipping...")
            continue
        
        print(f"Preparing {args.prefix}_{split}.pt ...")
        df_split = data[data['split'] == split]
        split_ids, split_y = list(df_split['system_id']), list(df_split['pK'])
        split_data = utils.GraphDataset(root='data', dataset=f'{args.prefix}_{split}', ids=split_ids, y=split_y, graphs_dict=graphs_dict)
