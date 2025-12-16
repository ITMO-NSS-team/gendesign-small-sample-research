''' '''
import ctopo

import os
import numpy as np
import pandas as pd
from tqdm import tqdm 

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdmd, MACCSkeys
from rdkit import DataStructs
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

from collections import defaultdict

# read ligands
df = pd.read_csv('data/Gd/final/merged_reviews_mapped.csv')
Ls = [ctopo.Ligand(smi) for smi in df['smiles_mapped']]
target = df['lgK']

# dict of reprs
reprs = {
    'ligand': lambda x: x._mol,
    'topo': lambda x: x.get_topology(True, False),
    'topo_da': lambda x: x.get_topology(True, True),
    'skl': lambda x: x.get_skeleton(True, False, False, False),
    'skl_da': lambda x: x.get_skeleton(True, True, False, False),
    'skl_da_skl': lambda x: x.get_skeleton(True, True, True, False),
    'skl_da_bonds': lambda x: x.get_skeleton(True, True, False, True),
    'skl_da_skl_bonds': lambda x: x.get_skeleton(True, True, True, True),
    'cmplx': lambda x: x.get_complexophore(True, False, False, False, False, False),
    'cmplx_da': lambda x: x.get_complexophore(True, True, False, False, False, False),
    'cmplx_da_bonds': lambda x: x.get_complexophore(True, True, False, False, True, False),
    'cmplx_da_sub': lambda x: x.get_complexophore(True, True, False, True, False, False),
    'cmplx_da_sub_bonds': lambda x: x.get_complexophore(True, True, False, True, True, False),
    'cmplx_full': lambda x: x.get_complexophore(True, True, True, True, True, True),
}

reprs_merged = {
    'skl_da': lambda x: x.get_skeleton(True, True, False, False),
    'cmplx_sub': lambda x: x.get_complexophore(True, False, False, True, False, False),
}

output_dir = 'data/Gd/fingerprints'

# =====================================
# Functions for computing descriptors
# =====================================

def compute_mol_descriptors(mol):
    if mol is not None:
        out = {}
        for name, fn in Descriptors._descList:
            out[name] = fn(mol)
        return out
    else:
        return {name: np.nan for name, _ in Descriptors._descList}

def compute_morgan_fingerprint(mol, radius=2, nBits=2048):
    if mol is not None:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits))
    else:
        return [None] * nBits

def compute_atom_pair_fingerprint(mol, nBits=2048):
    if mol is not None:
        return np.array(rdmd.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits))
    else:
        return [None] * nBits

def compute_maccs_keys(mol, nBits=166):
    if mol is not None:
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    else:
        return [None] * nBits

def compute_rdkit_fingerprint(mol, nBits=2048, minPath=1, maxPath=7):
    if mol is not None:
        return np.array(Chem.RDKFingerprint(mol, fpSize=nBits, minPath=minPath, maxPath=maxPath))
    else:
        return [None] * nBits

funcs_fps = {
    'morgan': compute_morgan_fingerprint,
    'atom_pair': compute_atom_pair_fingerprint,
    'maccs': compute_maccs_keys,
    'rdkit': compute_rdkit_fingerprint,
}

# ====================
# Additional functions
# ====================

def save_df(df, name, label, folder):
    if label: 
        output_file_path = os.path.join(output_dir, f'{folder}/Gd_ctopo_fp_{name}_{label}.csv')
    else:
        output_file_path = os.path.join(output_dir, f'{folder}/Gd_ctopo_fp_{name}.csv')   
    df.to_csv(output_file_path, index=False)

def delete_duplicates(df):
    bit_cols = [c for c in df.columns if c != 'lgK']
    df['lgK'] = df.groupby(bit_cols)['lgK'].transform('median')
    df = df.drop_duplicates(subset=bit_cols, keep='first').reset_index(drop=True)
    return df

def _pair_min_tanimoto_indices(bitvects):
    n = len(bitvects)
    best_pair = (0, 1)
    best_sim = 1.0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[i + 1:])
        for k, sim in enumerate(sims, start=1):
            if sim < best_sim:
                best_sim = sim
                best_pair = (i, i + k)
    return best_pair[0], best_pair[1], best_sim

def select_dissimilar_smiles(mol_list, k=5):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mol_list]
    n = len(fps)
    k = min(k, n)
    if n == 0 or k == 0:
        return [], []
    if n <= k:
        idxs = list(range(n))
        return idxs, [mol_list[i] for i in idxs]
    if k == 1:
        return [0], [mol_list[0]]

    # least-similar pair
    i, j, _ = _pair_min_tanimoto_indices(fps)
    selected = [i, j]

    # minimizing the max similarity to current set
    while len(selected) < k:
        best_cand, best_obj = None, float("inf")
        for c in range(n):
            if c in selected:
                continue
            max_sim = max(DataStructs.TanimotoSimilarity(fps[c], fps[s]) for s in selected)
            if (max_sim < best_obj) or (max_sim == best_obj and (best_cand is None or c < best_cand)):
                best_obj = max_sim
                best_cand = c
        selected.append(best_cand)
    return selected, [mol_list[i] for i in selected]

def select_indices_by_topology(Ls, smiles_series, topo_func, k):
    topo_mols = [topo_func(L) for L in Ls]
    topo_keys = [Chem.MolToSmiles(m, canonical=True) for m in topo_mols]

    from collections import defaultdict
    groups = defaultdict(list)
    for idx, key in enumerate(topo_keys):
        groups[key].append(idx)

    selected_global = set()
    records = []

    for key, idxs in groups.items():
        if len(idxs) <= k:
            for gi in idxs:
                selected_global.add(gi)
                records.append((key, gi, smiles_series.iloc[gi]))
        else:
            mols_group = [Ls[gi]._mol for gi in idxs]
            local_keep, _ = select_dissimilar_smiles(mols_group, k=k)
            keep_global = [idxs[j] for j in local_keep]
            for gi in keep_global:
                selected_global.add(gi)
                records.append((key, gi, smiles_series.iloc[gi]))

    selected_idx_sorted = sorted(selected_global)
    sel_map_df = pd.DataFrame(records, columns=['topology_key', 'global_index', 'smiles_mapped'])
    return selected_idx_sorted, sel_map_df

def fix_Hs(mol, name, fp_name):
    if (name in ['cmplx_da_sub', 'cmplx_da_sub_bonds', 'cmplx_full']) and (fp_name == 'atom_pair'):
        m = Chem.Mol(mol)
        m = Chem.RemoveHs(m, sanitize=False)
        return m
    else:
        return mol

# =========================================
# Compute molecular descriptors fot ligands
# =========================================

lig_mol_descriptors = [compute_mol_descriptors(L._mol) for L in tqdm(Ls, desc="Computing ligand mol descriptors")]
lig_mol_descriptors_df = pd.DataFrame(lig_mol_descriptors)

# ============================
# Create fingerprints datasets
# ============================

for fp_name, fp_func in funcs_fps.items():
    print(f'Working with fps: {fp_name}')
    for name, func in reprs.items():
        fingerprints = []
        for L in tqdm(Ls, desc=f"Building fps: {name}"):
            mol = func(L)
            fingerprints.append(fp_func(fix_Hs(mol, name, fp_name)))
        fp_df = pd.DataFrame(fingerprints, columns=[f'bit_{i}' for i in range(len(fingerprints[0]))])
        # for morgan save dataset without ligand's molecular descriptors
        if fp_name == 'morgan':
            fp_df['lgK'] = target.values
            save_df(fp_df, name, None, 'ctopo_dupl')

            if ("ligand" in name) or ("cmplx" in name):
                fp_df_wo_dupl = delete_duplicates(fp_df)
                save_df(fp_df_wo_dupl, name, 'wo_dupl', 'ctopo_wo_dupl')
        # mol descs for ligand + different fps
        merged_df = pd.concat([lig_mol_descriptors_df, fp_df], axis=1)
        merged_df['lgK'] = target.values
        save_df(merged_df, name, f'moldesc_{fp_name}', f'ctopo_moldesc/{fp_name}')

# selected by topology
K_LIST = [5, 10]
topo_func = reprs['topo']

for k in K_LIST:
    selected_idx_sorted, sel_map_df = select_indices_by_topology(Ls, df['smiles_mapped'], topo_func, k=k)
    sel_map_df.to_csv(f'data/Gd/final/merged_reviews_mapped_selected_by_topo_k{k}.csv', index=False)

    Ls_sel = [Ls[i] for i in selected_idx_sorted]
    target_sel = target.iloc[selected_idx_sorted].reset_index(drop=True)

    lig_mol_descriptors_sel = [compute_mol_descriptors(L._mol) for L in tqdm(Ls_sel, desc=f"Computing ligand mol descriptors (by topo k={k})")]
    lig_mol_descriptors_sel_df = pd.DataFrame(lig_mol_descriptors_sel)

    for fp_name, fp_func in funcs_fps.items():
        print(f'Working with fps: {fp_name}')
        for name, func in reprs.items():
            if name == 'topo':
                continue  
            fingerprints_sel = []
            for L in tqdm(Ls_sel, desc=f"Building fps by topo {k}: {name}"):
                mol = func(L)
                fingerprints_sel.append(fp_func(fix_Hs(mol, name, fp_name)))
            fp_sel_df = pd.DataFrame(fingerprints_sel, columns=[f'bit_{i}' for i in range(len(fingerprints_sel[0]))])
            
            # for morgan save dataset without ligand's molecular descriptors    
            if fp_name == 'morgan':
                if ("ligand" in name) or ("cmplx" in name):    
                    fp_sel_df['lgK'] = target_sel.values
                    fp_sel_df_wo_dupl = delete_duplicates(fp_sel_df)
                    save_df(fp_sel_df_wo_dupl, name, f"by_topo_{k}", f'ctopo_by_topo_{k}')
            
            # mol descs for ligand + different fps
            merged_sel_df = pd.concat([lig_mol_descriptors_sel_df, fp_sel_df], axis=1)
            merged_sel_df['lgK'] = target_sel.values
            save_df(merged_sel_df, name, f"moldesc_{fp_name}_by_topo_{k}", f'ctopo_moldesc_by_topo_{k}/{fp_name}')


# merged
fps = {}
for name, func in reprs_merged.items():
    fingerprints = []
    for L in tqdm(Ls, desc=f"Building merged fps: {name}"):
        mol = func(L)
        fingerprints.append(compute_morgan_fingerprint(mol))
    fps[name] = pd.DataFrame(fingerprints, columns=[f"{name}_bit_{i}" for i in range(2048)])

merged_df = pd.concat([fps["skl_da"], fps["cmplx_sub"]], axis=1)
merged_df["lgK"] = target.values

merged_df_wo_dupl = delete_duplicates(merged_df)

names_in_order = [name for name, _ in reprs_merged.items()]
merged_name = '_'.join(names_in_order)
save_df(merged_df_wo_dupl, merged_name, 'merged', 'ctopo_merged')