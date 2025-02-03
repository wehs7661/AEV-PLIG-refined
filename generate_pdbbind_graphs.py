import pandas as pd
import pickle
import torch
import torchani
import torchani_mod
import qcelemental as qcel
import numpy as np
from tqdm import tqdm
import os
from rdkit import Chem


def elements_to_atomicnums(elements):
    """
    Convert element symbols to atomic numbers.

    Parameters
    ----------
    elements: Iterable
        Iterable object with lement symbols

    Returns
    -------
    np.ndarray
        Array of atomic numbers
    """
    atomicnums = np.zeros(len(elements), dtype=int)

    for idx, e in enumerate(elements):
        atomicnums[idx] = qcel.periodictable.to_Z(e)

    return atomicnums


def LoadMolasDF(mol):
# This function converts the input ligand (.MOL file) into a pandas DataFrame with the ligand atom position in 3D (X,Y,Z)
    
    # load the ligand 
    m = mol

    atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)

    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX","ATOM_TYPE","X","Y","Z"]
    
    return df


def LoadPDBasDF(PDB, atom_keys):
# This function converts a protein PDB file into a pandas DataFrame with the protein atom position in 3D (X,Y,Z)

    prot_atoms = []
    
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])
                
    f.close()
    
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return df


def GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map):
    """
    Calculate AEVs for molecule atoms based on protein atoms in a complex.
    Hydrogens in both the molecule and protein are ignored and behind the
    curtains molecule atoms are encoded as carbon to use torchani_mod code.
    
    Parameters
    ----------
    protein_path: string, path to protein file (.pdb)
    mol_path: string, path to molecule file (.mol2)
    atom_keys: dataframe of atom types in proteins
    radial_coefs: list of pytorch tensors of RcR, EtaR, RsR
    angular_coefs: list of pytorch tensors of RcA, Zeta, TsA, EtaA, RsA

    Returns
    -------
    Tensor of shape [1, A, L], where A is the number of atoms and L is the 
    appropriate length of AEV.
    
    Todo
    ----
    Assert that RcR > RcA
    """
    
    # Protein and ligand structure are loaded as pandas DataFrame
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)
    
    # Define AEV coeficients
    # Radial coefficients
    RcR = radial_coefs[0]
    EtaR = radial_coefs[1]
    RsR = radial_coefs[2]
    
    # Angular coefficients (Ga)
    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA = torch.tensor([1.0]) # Angular shift in GA
    EtaA = torch.tensor([1.0])
    RsA = torch.tensor([1.0])  # Radial shift in GA
    
    # Reduce size of Target df to what we need based on radial cutoff RcR
    distance_cutoff = RcR + 0.1
    
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    Target = Target.merge(atom_map, on='ATOM_TYPE', how='left')
    
    # Create tensors of atomic numbers and coordinates of molecule atoms and 
    # protein atoms combined. Encode molecule atoms as hydrogen
    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len)*6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64)
    atomicnums = atomicnums.unsqueeze(0)
    
    coordinates = pd.concat([Ligand[['X','Y','Z']], Target[['X','Y','Z']]])
    coordinates = torch.tensor(coordinates.values)
    coordinates = coordinates.unsqueeze(0)
    
    # Use torchani_mod to calculate AEVs
    atom_symbols = []
    for i in range(1, 23):
        atom_symbols.append(qcel.periodictable.to_symbol(i))
        
    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, 
                                EtaA, Zeta, RsA, TsA, len(atom_symbols))
    
    SC = torchani.SpeciesConverter(atom_symbols)
    sc = SC((atomicnums, coordinates))
    
    aev = AEVC.forward((sc.species, sc.coordinates), mol_len)
    
    # find indices of columns to keep
    # keep all radial terms
    n = len(atom_symbols)
    n_rad_sub = len(EtaR)*len(RsR)
    indices = list(np.arange(n*n_rad_sub))
    
    return Ligand, aev.aevs.squeeze(0)[:mol_len,indices]


def GetMolAEVs_rad(protein_path, mol, atom_keys, radial_coefs):
    """
    Calculate AEVs for molecule atoms based on protein atoms in a complex.
    Hydrogens in both the molecule and protein are ignored and behind the
    curtains molecule atoms are encoded as carbon to use torchani_mod code.
    
    Parameters
    ----------
    protein_path: string, path to protein file (.pdb)
    mol_path: string, path to molecule file (.mol2)
    atom_keys: dataframe of atom types in proteins
    radial_coefs: list of pytorch tensors of RcR, EtaR, RsR
    angular_coefs: list of pytorch tensors of RcA, Zeta, TsA, EtaA, RsA

    Returns
    -------
    Tensor of shape [1, A, L], where A is the number of atoms and L is the 
    appropriate length of AEV.
    
    Todo
    ----
    Assert that RcR > RcA
    """
    
    # Protein and ligand structure are loaded as pandas DataFrame
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)
    
    # Define AEV coeficients
    # Radial coefficients
    RcR = radial_coefs[0]
    EtaR = radial_coefs[1]
    RsR = radial_coefs[2]
    
    # Angular coefficients (Ga)
    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA = torch.tensor([1.0]) # Angular shift in GA
    EtaA = torch.tensor([1.0])
    RsA = torch.tensor([1.0])  # Radial shift in GA
    
    # Reduce size of Target df to what we need based on radial cutoff RcR
    distance_cutoff = RcR + 0.1
    
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    # Instead of all these different atom types we only use atom symbol
    Target["ATOM_SYMBOL"] = [i.split(";")[0] for i in Target["ATOM_TYPE"]]
    
    # Create tensors of atomic numbers and coordinates of molecule atoms and 
    # protein atoms combined. Encode molecule atoms as hydrogen
    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len)*6, elements_to_atomicnums(Target['ATOM_SYMBOL']))
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64)
    atomicnums = atomicnums.unsqueeze(0)
    
    coordinates = pd.concat([Ligand[['X','Y','Z']], Target[['X','Y','Z']]])
    coordinates = torch.tensor(coordinates.values)
    coordinates = coordinates.unsqueeze(0)
    
    # Use torchani_mod to calculate AEVs
    atom_symbols = ['C','N','O','S']
    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, 
                                EtaA, Zeta, RsA, TsA, len(atom_symbols))
    
    SC = torchani.SpeciesConverter(atom_symbols)
    sc = SC((atomicnums, coordinates))
    
    aev = AEVC.forward((sc.species, sc.coordinates), mol_len)

    # find indices of columns to keep
    # keep all radial terms
    n = len(atom_symbols)
    n_rad_sub = len(EtaR)*len(RsR)
    indices = list(np.arange(n*n_rad_sub))
    
    return Ligand, aev.aevs.squeeze(0)[:mol_len,indices]


def one_of_k_encoding(x, allowable_set):
    """
    taken from https://github.com/thinng/GraphDTA

    function which one hot encodes x w.r.t. allowable_set and x has to be in allowable_set

    x:
        element from allowable_set
    allowable_set: list
        list of elements x is from
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, features=["atom_symbol",
                                  "num_heavy_atoms", 
                                  "total_num_Hs", 
                                  "explicit_valence", 
                                  "is_aromatic", 
                                  "is_in_ring"]):
    # Computes the ligand atom features for graph node construction
    # The standard features are the following:
    # atom_symbol = one hot encoding of atom symbol
    # num_heavy_atoms = # of heavy atom neighbors
    # total_num_Hs = # number of hydrogen atom neighbors
    # explicit_valence = explicit valence of the atom
    # is_aromatic = boolean 1 - aromatic, 0 - not aromatic
    # is_in_ring = boolean 1 - is in ring, 0 - is not in ring

    feature_list = []
    if "atom_symbol" in features:
        feature_list.extend(one_of_k_encoding(atom.GetSymbol(),['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']))
    if "num_heavy_atoms" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))
    if "total_num_Hs" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))
    if "explicit_valence" in features: #-NEW ADDITION FOR PLIG
        feature_list.append(atom.GetExplicitValence())
    if "is_aromatic" in features:
        
        if atom.GetIsAromatic():
            feature_list.append(1)
        else:
            feature_list.append(0)
    if "is_in_ring" in features:
        if atom.IsInRing():
            feature_list.append(1)
        else:
            feature_list.append(0)
    
    return np.array(feature_list)


def mol_to_graph(mol, mol_df, aevs, extra_features=["atom_symbol",
                                                    "num_heavy_atoms", 
                                                    "total_num_Hs", 
                                                    "explicit_valence",
                                                    "is_aromatic",
                                                    "is_in_ring"]):

    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0
    
    # Generate nodes
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            feature = np.append(atom_features(atom), aevs[aev_idx,:])
            features.append(feature)
            counter += 1
    
    #Generate edges
    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            bond_type = one_of_k_encoding(bond.GetBondType(),[1,12,2,3])
            bond_type = [float(b) for b in bond_type]
            edge1 = [idx_to_idx[idx1], idx_to_idx[idx2]]
            edge1.extend(bond_type)
            edge2 = [idx_to_idx[idx2], idx_to_idx[idx1]]
            edge2.extend(bond_type)
            edges.append(edge1)
            edges.append(edge2)
    
    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic', 'double', 'triple'])
    df = df.sort_values(by=['atom1','atom2'])
    
    edge_index = df[['atom1','atom2']].to_numpy().tolist()
    edge_attr = df[['single','aromatic','double','triple']].to_numpy().tolist()
    
    
    return len(mol_df), features, edge_index, edge_attr

'''
Load data
'''
data = pd.read_csv("data/pdbbind_processed.csv", index_col=0)

'''
Generate for all complexes: ANI-2x with 22 atom types. Only 2-atom interactions
'''
print("The number of data points is ", len(data))

atom_keys = pd.read_csv("data/PDB_Atom_Keys.csv", sep=",")
atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
atom_map[1] = list(np.arange(len(atom_map)) + 1)
atom_map = atom_map.rename(columns={0:"ATOM_TYPE", 1:"ATOM_NR"})

# Radial coefficients: ANI-2x
RcR = 5.1 # Radial cutoff
EtaR = torch.tensor([19.7]) # Radial decay
RsR = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68, 
                    2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83]) # Radial shift
radial_coefs = [RcR, EtaR, RsR]

mol_graphs = {}

failed_list = []
failed_after_reading = []

for i, pdb in tqdm(enumerate(data["PDB_code"])):
    if data["refined"][i]:
        folder = "data/pdbbind/refined-set/"
    else:
        folder = "data/pdbbind/general-set/"
    
    mol_path = os.path.join(folder, pdb, f'{pdb}_ligand.mol2')
    mol = Chem.MolFromMol2File(mol_path)
    
    if mol is None:
        print("can't read molecule structure:", pdb)
        failed_list.append(pdb)
        continue
    else:
        mol = Chem.AddHs(mol, addCoords=True)

    try:
        protein_path = os.path.join(folder, pdb, f'{pdb}_protein.pdb')
        
        mol_df, aevs = GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map)
        graph = mol_to_graph(mol, mol_df, aevs)
        mol_graphs[pdb] = graph
        

    except ValueError as e:
        print(e)
        failed_after_reading.append(pdb)
        continue

print(len(failed_list), len(failed_after_reading))


#save the graphs to use as input for the GNN models
output_file_graphs = "data/pdbbind.pickle"
with open(output_file_graphs, 'wb') as handle:
    pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)


