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
from aev_plig import utils, nn_utils


def initialize(args):
    parser = argparse.ArgumentParser(
        description="This CLI process pickled graphs and create PyTorch data ready for training."
    )
    parser.add_argument(
        "-pg",
        "--pickled_graphs",
        type=str,
        nargs='+',
        required=True,
        help="The paths to the pickled graphs. The pickled graphs should be in the format of .pkl or .pickle."
    )
    parser.add_argument(
        "-c",
        "--csv_files",
        type=str,
        nargs='+',
        required=True,
        help="The paths to the input processed CSV files, each corresponding to a pickled graph. The order of the CSV files \
            should match the order of the pickled graphs. The CSV files should contain the columns 'system_id', 'protein_path', \
            'ligand_path', 'pK', and 'split'. The 'system_id' column should contain the same IDs as the pickled graphs."
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
    """
    Parse the filter string into column, operator, and value.
    The filter string should be in the following format: <column> <operator> <value>.
    For example: 'max_tanimoto_schrodinger < 0.9'.

    Parameters
    ----------
    filter_str : str
        The filter string to parse.
    
    Returns
    -------
    column : str
        The column name to filter on.
    operator : str
        The operator to use for filtering. Supported operators are: <, <=, >, >=, ==, !=.
    value : str or float
        The value to compare against. This will be converted to a float if possible.
    """
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
    """
    Apply the filter to the DataFrame based on the column, operator, and value.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    column : str
        The column name to filter on.
    operator : str
        The operator to use for filtering. Supported operators are: <, <=, >, >=, ==, !=.
    value : str or float
        The value to compare against. This will be converted to a float if possible.
    
    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
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
    print(f"Loading pickled graphs from {args.pickled_graphs}...")
    graphs_dict = {}
    for pickled_graph in args.pickled_graphs:
        with open(pickled_graph, "rb") as f:
            graphs = pickle.load(f)
        graphs_dict.update(graphs)

    # 2. Load the processed CSV files
    data_to_merge = []
    for csv_file in args.csv_files:
        print(f"Processing {csv_file}...")
        csv_data = pd.read_csv(csv_file)
        if args.filters:
            for filter_str in args.filters:
                column, operator, value = parse_filter(filter_str)
                csv_data = apply_filter(csv_data, column, operator, value)
        csv_data = csv_data[['system_id', 'pK', 'split']]
        
        data_to_merge.append(csv_data)
    data = pd.concat(data_to_merge, ignore_index=True)
    print(f'\nNumber of training entries: {len(data[data["split"] == "train"])}')
    print(f'Number of validation entries: {len(data[data["split"] == "validation"])}')
    print(f'Number of test entries: {len(data[data["split"] == "test"])}')
    print(f'Number of other entries: {len(data[data["split"] == "others"])}\n')

    # 3. Create PyTorch data
    splits = ['train', 'validation', 'test']
    for split in splits:
        if split not in data['split'].unique():
            print(f"\nSplit '{split}' not found in the dataset. Skipping ...")
            continue
        
        df_split = data[data['split'] == split]
        split_ids, split_y = list(df_split['system_id']), list(df_split['pK'])
        split_data = nn_utils.GraphDataset(root='data', dataset=f'{args.prefix}_{split}', ids=split_ids, y=split_y, graphs_dict=graphs_dict)

    print(f"Elapsed time: {utils.format_time(time.time() - t0)}")
