import pandas as pd
import pickle
from utils import GraphDataset

"""
Load graphs
"""
print("loading graph from pickle file for pdbbind2020")
with open("data/pdbbind.pickle", 'rb') as handle:
    pdbbind_graphs = pickle.load(handle)

print("loading graph from pickle file for BindingNet")
with open("data/bindingnet.pickle", 'rb') as handle:
    bindingnet_graphs = pickle.load(handle)

print("loading graph from pickle file for BindingDB")
with open("data/bindingdb.pickle", 'rb') as handle:
    bindingdb_graphs = pickle.load(handle)

graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}


"""
Generate data for training for <0.9 Tanimoto to Schrodinger/Merck
"""

'''
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]
pdbbind = pdbbind[['unique_id','pK','split']]

bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
bindingnet = bindingnet.rename(columns={'-logAffi': 'pK','unique_identify':'unique_id'})[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingnet['split'] = 'train'
bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.9]
bindingnet = bindingnet[['unique_id','pK','split']]

bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
bindingdb = bindingdb[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingdb['split'] = 'train'
bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9]
bindingdb = bindingdb[['unique_id','pK','split']]

# combine pdbbind2020, bindingnet, and bindingdb index sets
data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

# make data PyTorch Geometric ready
print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)
'''


"""
Generate data for training for <0.8 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.8]
pdbbind = pdbbind[['unique_id','pK','split']]

bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
bindingnet = bindingnet.rename(columns={'-logAffi': 'pK','unique_identify':'unique_id'})[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingnet['split'] = 'train'
bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.8]
bindingnet = bindingnet[['unique_id','pK','split']]

bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
bindingdb = bindingdb[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingdb['split'] = 'train'
bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.8]
bindingdb = bindingdb[['unique_id','pK','split']]

# combine pdbbind2020, bindingnet, and bindingdb index sets
data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim80_fep_benchmark'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

# make data PyTorch Geometric ready
print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


"""
Generate data for training for <0.7 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.7]
pdbbind = pdbbind[['unique_id','pK','split']]

bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
bindingnet = bindingnet.rename(columns={'-logAffi': 'pK','unique_identify':'unique_id'})[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingnet['split'] = 'train'
bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.7]
bindingnet = bindingnet[['unique_id','pK','split']]

bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
bindingdb = bindingdb[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingdb['split'] = 'train'
bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.7]
bindingdb = bindingdb[['unique_id','pK','split']]

# combine pdbbind2020, bindingnet, and bindingdb index sets
data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim70_fep_benchmark'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

# make data PyTorch Geometric ready
print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


"""
Generate data for training for <0.6 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.6]
pdbbind = pdbbind[['unique_id','pK','split']]

bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
bindingnet = bindingnet.rename(columns={'-logAffi': 'pK','unique_identify':'unique_id'})[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingnet['split'] = 'train'
bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.6]
bindingnet = bindingnet[['unique_id','pK','split']]

bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
bindingdb = bindingdb[['unique_id','pK','max_tanimoto_fep_benchmark']]
bindingdb['split'] = 'train'
bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.6]
bindingdb = bindingdb[['unique_id','pK','split']]

# combine pdbbind2020, bindingnet, and bindingdb index sets
data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim60_fep_benchmark'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

# make data PyTorch Geometric ready
print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)