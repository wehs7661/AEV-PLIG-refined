import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def generate_fingerprints(df, col_name):
    """
    Generate fingerprints for the ligands in the given CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ligand data.
    col_name : str
        The name of the column in the CSV file that contains the SDF file paths.
    
    Returns
    -------
    fingerprints : list
        A list of generated fingerprints for each ligand in the CSV file.
    valid_indices : list
        A list of indices for which the fingerprints were successfully generated.
    """
    fp_gen = GetMorganGenerator(radius=3)
    fingerprints, valid_indices = [], []
    
    for index, row in tqdm(df.iterrows()):
        suppl = Chem.SDMolSupplier(row[col_name], removeHs=True)
        lig = suppl[0]
        if lig is None:
            print(f"Warning: RDKit returned None for {row[col_name]} (index {index}). Skipping ...")
            continue
        fingerprints.append(fp_gen.GetSparseCountFingerprint(lig))
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
    for i in range(len(fps1)):
        max_similarities.append(BulkTanimotoSimilarity(fps1[i], fps2))
    max_similarities = np.array(max_similarities)
    max_similarities = max_similarities.max(axis=1)

    return max_similarities
