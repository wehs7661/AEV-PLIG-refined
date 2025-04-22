import os
import sys
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def get_atom_types_from_sdf(sdf_file):
    """
    Get the atom types from an SDF file.

    Parameters
    ----------
    sdf_file : str
        Path to the SDF file.
    
    Returns
    -------
    atom_types : list
        List of atom types.
    """
    suppl = Chem.SDMolSupplier(sdf_file)
    atom_types = []
    for mol in suppl:
        if mol is not None:
            for atom in mol.GetAtoms():
                atom_types.append(atom.GetSymbol())
    
    atom_types = list(set(atom_types))
    return atom_types


def get_atom_types_from_sdf_parallelized(paths):
    with mp.Pool(initializer=lambda:os.sched_setaffinity(0, set(range(mp.cpu_count())))) as pool:
        results = list(
            tqdm(
                pool.imap(get_atom_types_from_sdf, paths),
                total=len(paths),
                desc="Getting atom types from SDF",
                file=sys.__stderr__,
            )
        )
    return results


def generate_fingerprint(sdf_file):
    """
    Process a single SDF file path to generate its ligand fingerprint.
    
    Parameters
    ----------
    sdf_file : str
        Path to the SDF file.
    
    Returns
    -------
    fingerprint : rdkit.DataStructs.cDataStructs.ULongSparseIntVect
        The generated fingerprint for the ligand in the SDF file.
    """
    fp_gen = GetMorganGenerator(radius=3)
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=True)
    lig = suppl[0]
    if lig is None:
        print(f"Warning: RDKit returned None for {sdf_file}. Skipping ...")
        return None
    fingerprint = fp_gen.GetSparseCountFingerprint(lig)
    return fingerprint


def generate_fingerprints_parallelized(df, col_name):
    """
    Generate fingerprints for ligands in a pandas DataFrame using parallel processing.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ligand data.
    col_name : str
        The name of the column in the DataFrame that contains the SDF file paths.
    
    Returns
    -------
    fingerprints : list
        A list of generated fingerprints for each ligand in the DataFrame.
    valid_indices : list
        A list of indices for which the fingerprints were successfully generated.
    """
    sdf_paths = df[col_name].tolist()
    with Pool(initializer=lambda: os.sched_setaffinity(0, set(range(cpu_count())))) as pool:
        results = list(tqdm(pool.imap(generate_fingerprint, sdf_paths),
                            total=len(sdf_paths),
                            file=sys.__stderr__,
                            desc="Generating fingerprints"))
    fingerprints, valid_indices = [], []
    for index, fingerprint in enumerate(results):
        if fingerprint is not None:
            fingerprints.append(fingerprint)
            valid_indices.append(index)
    
    return fingerprints, valid_indices
