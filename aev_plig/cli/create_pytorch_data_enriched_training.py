import pandas as pd
import numpy as np
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

print("loading graph from pickle file for enrichment data")
with open("data/fep_benchmark_graphs.pickle", 'rb') as handle:
    fep_benchmark_graphs = pickle.load(handle)


graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs, **fep_benchmark_graphs}

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

R = 1.987e-3
T = 297

"""
Generate data for enriched training for <0.9 Tanimoto to Schrodinger/Merck

Random 1
"""
"""
# 1 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"] == 1]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich1_random1'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 2 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([1,2])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich2_random1'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 3 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([1,2,3])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich3_random1'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 4 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([1,2,3,4])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich4_random1'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 5 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([1,2,3,4,5])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich5_random1'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)
"""

"""
Generate data for enriched training for <0.9 Tanimoto to Schrodinger/Merck

Random 2
"""

"""
# 1 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"] == 6]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich1_random2'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 2 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([6,7])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich2_random2'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 3 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([6,7,8])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich3_random2'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 4 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([6,7,8,9])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich4_random2'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 5 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([6,7,8,9,10])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich5_random2'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)

"""

"""
Generate data for enriched training for <0.9 Tanimoto to Schrodinger/Merck

Random 3
"""
"""
# 1 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"] == 11]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich1_random3'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 2 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([11,12])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich2_random3'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


# 3 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([11,12,13])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich3_random3'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)

"""

# 4 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([11,12,13,14])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich4_random3'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)

"""
# 5 enrichment ligand
enrichment = pd.read_csv("data/fep_benchmark_enriched.csv", index_col=0)
enrichment = enrichment[enrichment["enumeration"].isin([11,12,13,14,15])]
enrichment["pK"] = -enrichment["Exp. dG (kcal/mol)"]/(R*T*np.log(10))
enrichment = enrichment[['unique_id','pK']]
enrichment['split'] = 'train'

data = pd.concat([pdbbind, bindingnet, bindingdb, enrichment], ignore_index=True)
print(data[['split']].value_counts())

dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_enrich5_random3'

df = data[data['split'] == 'train']
train_ids, train_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'valid']
valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

df = data[data['split'] == 'test']
test_ids, test_y = list(df['unique_id']), list(df['pK'])

print('preparing ', dataset + '_train.pt in pytorch format!')
train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_valid.pt in pytorch format!')
valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

print('preparing ', dataset + '_test.pt in pytorch format!')
test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)

"""