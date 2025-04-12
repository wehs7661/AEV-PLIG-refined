import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from multiprocessing import Pool, cpu_count


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

def calc_max_tanimoto_similarity(fps1, fps2):
    """
    Given two sets of fingerprints, calculate the maximum Tanimoto similarity
    between each fingerprint in the first set and all fingerprints in the second set.
    
    Parameters
    ----------
    fps1 : list
        The first set of fingerprints.
    fps2 : list
        The second set of fingerprints.
    
    Returns
    -------
    max_similarities : list
        A list of maximum Tanimoto similarities against the second set of fingerprints for each
        fingerprint in the first set.
    """
    max_similarities = []
    for i in tqdm(range(len(fps1)), file=sys.__stderr__, desc="Calculating Tanimoto similarities"):
        max_similarities.append(BulkTanimotoSimilarity(fps1[i], fps2))
    max_similarities = np.array(max_similarities)
    max_similarities = max_similarities.max(axis=1)

    return max_similarities
