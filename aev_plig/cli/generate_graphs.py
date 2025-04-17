import os
import sys
import time
import torch
import pickle
import argparse
import torchani
import datetime
import aev_plig
import torchani_mod
import numpy as np
import pandas as pd
import qcelemental as qcel
from tqdm import tqdm
from rdkit import Chem
from aev_plig import utils
from aev_plig.data import data_dir
from multiprocessing import Pool, cpu_count


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This CLI generates graphs for the input dataset.'
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        required=True,
        help="The path to the input processed CSV file. The CSV file should at least have columns including 'system_id',\
            'protein_path', and 'ligand_path'."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="graphs.pickle",
        help="The path to the output pickle file for the generated graphs. Default is graphs.pickle."
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="generate_graphs.log",
        help="The path to the log file. Default is generate_graphs.log."
    )

    args = parser.parse_args(args)
    return args


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
    """
    Converts a protein PDB file into a pandas DataFrame having columns including
    ATOM_INDEX, ATOM_TYPE, X, Y, and Z. Note that this function filters out hydrogen atoms
    and do not consider HETATM records.

    Parameters
    ----------
    PDB: str
        Path to the PDB file.
    atom_keys: pd.DataFrame
        DataFrame containing atom types and their corresponding keys.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ATOM_INDEX, ATOM_TYPE, X, Y, and Z.
    """
    prot_atoms = []
    auto_serial = 100000  # Upon this number, we assume contiguous indexing
    max_seen_index = 0  # track max index for all atoms (including hydrogen)

    with open(PDB) as f:
        for i, line in enumerate(f):
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            index_str = line[6:11].strip()

            if index_str.isdigit():
                atom_index = int(index_str)
                if atom_index > max_seen_index:
                    max_seen_index = atom_index
            else:
                # print(max_atom_index, line)
                assert max_seen_index >= 99999, f"There exist atom indices with non-numeric values \
                    before the number of atoms exceeds 99999. (position: {i})"
                atom_index = auto_serial
                auto_serial += 1

            # Skip hydrogens when adding to prot_atoms
            is_hydrogen = (
                (len(atom_name) < 4 and atom_name[0] == "H") or
                (len(atom_name) == 4 and (atom_name[0] == "H" or atom_name[1] == "H"))
            )
            if is_hydrogen:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            pdb_atom = f"{resname}-{atom_name}"            
            prot_atoms.append([atom_index, pdb_atom, x, y, z])

    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = (
        df
        .merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')
        [["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]]
        .sort_values(by="ATOM_INDEX")
        .reset_index(drop=True)
    )

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


def process_row(args):
    """
    Process a single row of the DataFrame to generate a graph for the protein-ligand complex.
    
    Parameters
    ----------
    args: tuple
        Contains (index, row, atom_keys, atom_map, radial_coefs) where row is a Series from the DataFrame.
    
    Returns
    -------
    tuple
        (system_id, graph, failed, failed_after_reading, log_messages)
        - system_id: The identifier for the complex.
        - graph: The computed graph or None if failed.
        - failed: True if molecule loading failed, else False.
        - failed_after_reading: True if AEV/graph computation failed, else False.
    """
    index, row, atom_keys, atom_map, radial_coefs = args
    
    system_id = row["system_id"]
    protein_path = row["protein_path"]
    ligand_path = row["ligand_path"]
    ligand_ftype = ligand_path.split(".")[-1]
    
    # Load ligand
    if ligand_ftype == "mol2":
        mol = Chem.MolFromMol2File(ligand_path)
    elif ligand_ftype == "sdf":
        mol = Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
    else:
        print(f"The ligand file type {ligand_ftype} is not supported for system {system_id}. Only mol2 and sdf are supported.")
        return system_id, None, True, False
    
    if mol is None:
        print(f"Can't read molecule structure: {system_id}")
        return system_id, None, True, False
    else:
        if ligand_ftype == "mol2":
            mol = Chem.AddHs(mol, addCoords=True)
    
    # Compute AEVs and graph
    try:
        mol_df, aevs = GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map)
        graph = mol_to_graph(mol, mol_df, aevs)
        return system_id, graph, False, False
    except ValueError as e:
        print(f"ValueError in system {system_id}: {str(e)}")
        return system_id, None, False, True


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(args.log)
    sys.stderr = utils.Logger(args.log)

    print(f"Version of aev_plig: {aev_plig.__version__}")
    print(f"Command line: {' '.join(sys.argv)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current time: {datetime.datetime.now()}\n")
    
    # Step 1. Load data
    data = pd.read_csv(args.csv)
    print(f"Generating graphs for the training set {args.csv} ...")
    print(f"The number of data points: {len(data)}")

    # Step 2. Generate for all complexes: ANI-2x with 22 atom types. Only 2-atom interactions
    atom_keys = pd.read_csv(os.path.join(data_dir, "PDB_Atom_Keys.csv"), sep=",")
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0:"ATOM_TYPE", 1:"ATOM_NR"})

    # Step 3. Radial coefficients: ANI-2x
    RcR = 5.1 # Radial cutoff
    EtaR = torch.tensor([19.7]) # Radial decay
    RsR = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68, 
                        2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83]) # Radial shift
    radial_coefs = [RcR, EtaR, RsR]

    mol_graphs = {}
    failed_list = []
    failed_after_reading = []

    torch.set_num_threads(1)  # This is necessary or the parallelization would not help.
    pool_input = [(index, row, atom_keys, atom_map, radial_coefs) for index, row in data.iterrows()]
    with Pool(initializer=lambda: os.sched_setaffinity(0, set(range(cpu_count())))) as pool:
        for system_id, graph, failed, failed_after_reading_flag in tqdm(
            pool.imap(process_row, pool_input),
            total=len(data),
            file=sys.__stderr__,
            desc="Generating graphs",
        ):
            if failed:
                failed_list.append(system_id)
            elif failed_after_reading_flag:
                failed_after_reading.append(system_id)
            else:
                mol_graphs[system_id] = graph

    print("Number of failed molecules:", len(failed_list))
    print("Number of failed after reading:", len(failed_after_reading))

    # Save the graphs to use as input for the GNN models
    with open(args.output, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Elapsed time: {utils.format_time(time.time() - t0)}")
