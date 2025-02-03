import pandas as pd
import pickle
from utils import GraphDataset

"""
Load graphs
"""
print("loading graph from pickle file for pdbbind2020")
with open("data/pdbbind.pickle", 'rb') as handle:
    graphs_dict = pickle.load(handle)

'''
"""
Generate data for enriched training for <0.9 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]
pdbbind = pdbbind[['unique_id','pK','split']]

data = pdbbind.reset_index(drop=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_ligsim90_fep_benchmark'

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
Generate data for enriched training for <0.8 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.8]
pdbbind = pdbbind[['unique_id','pK','split']]

data = pdbbind.reset_index(drop=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_ligsim80_fep_benchmark'

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
Generate data for enriched training for <0.7 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.7]
pdbbind = pdbbind[['unique_id','pK','split']]

data = pdbbind.reset_index(drop=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_ligsim70_fep_benchmark'

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
Generate data for enriched training for <0.6 Tanimoto to Schrodinger/Merck
"""
pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.6]
pdbbind = pdbbind[['unique_id','pK','split']]

data = pdbbind.reset_index(drop=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_ligsim60_fep_benchmark'

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
Generate data for OOD Test
"""
oodtest = pd.read_csv("data/index_oodtest.csv", index_col=0)
oodtest = oodtest[oodtest["rare_atom_type"] == False]
oodtest = oodtest[oodtest["unspecified_bond"] == False]
oodtest = oodtest[oodtest["refined"] == True]
oodtest = oodtest[['PDB_code','-logKd/Ki','split_maxsep50']]
oodtest = oodtest.rename(columns={'PDB_code':'unique_id', 'split_maxsep50':'split', '-logKd/Ki':'pK'})

data = oodtest.reset_index(drop=True)
print(data['split'].value_counts())

dataset = 'oodtest'

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
