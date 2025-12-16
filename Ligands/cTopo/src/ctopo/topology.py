'''Functionality to define atom types in the ligand and to generate 
skeletons and topologies'''

#%% Imports

import itertools as _iter
import copy as _copy

from rdkit import Chem as _Chem

import ctopo.graphs as _graphs
import ctopo.visuals as _vis

import typing as _tp


#%% Annotation

def save_atom_properties(mol) -> None:
    ''' '''
    for a in mol.GetAtoms():
        a.SetProp('_atomic_symbol', a.GetSymbol())
        a.SetIntProp('_atomic_charge', a.GetFormalCharge())
        a.SetIntProp('_number_of_hydrogens', a.GetTotalNumHs())
    
    return


def save_bond_properties(mol) -> None:
    ''' '''
    for b in mol.GetBonds():
        bond_type = str(b.GetBondType())
        b.SetProp('_bond_type', bond_type)
        if str(b.GetBondType()) in ('DATIVE', 'DATIVEL', 'DATIVER'):
            bond_type = 'SINGLE'
        elif str(b.GetBondType()) in ('AROMATIC', ):
            bond_type = 'DOUBLE'
        b.SetProp('_reduced_bond_type', bond_type)
    
    return


def classify_ligand_atoms(mol) -> None:
    ''' '''
    # find DAs (from mapping)
    DAs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()]
    for idx in DAs:
        a = mol.GetAtomWithIdx(idx)
        a.SetProp('_ligand_atom_type', 'DA')
        a.SetAtomMapNum(0)
    # find skeleton
    skeleton = set()
    for da1, da2 in _iter.combinations(DAs, r = 2):
        paths = _graphs.get_all_shortest_paths(mol, da1, da2)
        for path in paths:
            skeleton.update(set(path))
    for idx in skeleton:
        a = mol.GetAtomWithIdx(idx)
        if a.HasProp('_ligand_atom_type') and a.GetProp('_ligand_atom_type') == 'DA':
            continue
        a.SetProp('_ligand_atom_type', 'SKL')
    # classify other atoms
    for a in mol.GetAtoms():
        if not a.HasProp('_ligand_atom_type'):
            a.SetProp('_ligand_atom_type', 'SUB')
    
    return


def annotate_ligand(mol) -> None:
    ''' '''
    save_atom_properties(mol)
    save_bond_properties(mol)
    classify_ligand_atoms(mol)
    
    return



#%% Skeletons

def simplify_molecular_graph(mol):
    '''Transforms atoms to dummies and bonds to SINGLE/DOUBLE'''
    for b in mol.GetBonds():
        b.SetBondType(_Chem.BondType.SINGLE)
    for a in mol.GetAtoms():
        a.SetAtomicNum(0)
        a.SetNumExplicitHs(0)
        a.SetIsAromatic(False)
        a.SetNumRadicalElectrons(0)
        a.SetFormalCharge(0)
    
    return


def get_ligand_skeleton(mol):
    ''' '''
    mol = _copy.deepcopy(mol)
    annotate_ligand(mol)
    simplify_molecular_graph(mol)
    # visual labels for DAs
    _vis.draw_skeletopo(mol)
    # atoms to remove
    remove_idxs = [a.GetIdx() for a in mol.GetAtoms()
                       if a.GetProp('_ligand_atom_type') == 'SUB']
    remove_idxs = sorted(remove_idxs, reverse=True)
    # remove
    edmol = _Chem.EditableMol(mol)
    for idx in remove_idxs:
        edmol.RemoveAtom(idx)
    mol = edmol.GetMol()
    _Chem.SanitizeMol(mol)
    
    return mol



#%% Topologies

def get_next_atom_to_remove(mol: _Chem.rdchem.Mol, ignore_cycles: bool = False,
                            bubble: bool = False) -> _tp.Optional[_Chem.rdchem.Atom]:
    '''Takes prepared ligand as input and returns non-donor atom with 2 neighbors
    
    Arguments:
        - mol (Chem.rdchem.Mol): molecule after applying `prepare_ligand`
        - ignore_cycles (bool): do not remove cyclic atoms if True
        - bubble (bool): linear linker with bonded neighbors
    
    Returns:
        Optional[Chem.rdchem.Atom]: first non-donor atom with 2 neighbors.
            Returns None in the absence of such atoms
    '''
    # check params
    if ignore_cycles and bubble:
        raise ValueError('Bubble regime is possible with `ignore_cycles=False` only')
    # search atom
    for a in mol.GetAtoms():
        if a.GetProp('_ligand_atom_type') == 'DA':
            continue
        if ignore_cycles and a.IsInRing():
            continue
        ns = a.GetNeighbors()
        if len(ns) != 2:
            continue
        ns_bonded = mol.GetBondBetweenAtoms(ns[0].GetIdx(), ns[1].GetIdx())
        if not bubble and ns_bonded:
            continue
        if not bubble or (bubble and ns_bonded):
            return a
    
    return None


def remove_atom(a: _Chem.rdchem.Atom, bubble: bool = False) -> _Chem.rdchem.Mol:
    '''Removes given atom from the corresponding molecule if it's a linear linker
    
    Arguments:
        - a (Chem.rdchem.Atom): atom to remove; must have exactly 2 neighbors
            and be a regular atom (non-DA)
        - bubble (bool): bubble atom is removed without additional bonding
    
    Returns:
        - Chem.rdchem.Mol: molecule with removed atom
    
    Raises:
        - ValueError: if given atom is DA or has more or less than 2 neighbors
    '''
    mol = a.GetOwningMol()
    # checks
    if a.GetProp('_ligand_atom_type') == 'DA':
        raise ValueError('The given atom is donor atom')
    ns = [n.GetIdx() for n in a.GetNeighbors()]
    if len(ns) != 2:
        raise ValueError(f'The given atom has {len(ns)} neighbors (2 is required for linear linker)')
    ns_bonded = mol.GetBondBetweenAtoms(ns[0], ns[1])
    if ns_bonded and not bubble:
        raise ValueError('The given atom is a bubble (neighbors are bonded)')
    if not ns_bonded and bubble:
        raise ValueError('The given atom is not a bubble (neighbors are not bonded)')
    # modify mol
    edm = _Chem.EditableMol(mol)
    if not bubble:
        edm.AddBond(ns[0], ns[1], _Chem.BondType.SINGLE)
    edm.RemoveAtom(a.GetIdx())
    mol = edm.GetMol()
    
    return mol


def get_ligand_topology(mol: _Chem.rdchem.Mol,
                        ignore_cycles: bool = False) -> _Chem.rdchem.Mol:
    '''Returns molecule representing topology of the given ligand
    
    Arguments:
        - mol (Chem.rdchem.Mol): RDKit molecule, donor atoms must be marked
            with non-zero atomic map numbers
        - ignore_cycles (bool): if True, does not remove ring atoms
    
    Returns:
        - Chem.rdchem.Mol: molecule describing ligand's topology
    '''
    # prepare skeleton
    topo = get_ligand_skeleton(mol)
    for b in topo.GetBonds():
        b.SetBondType(_Chem.BondType.SINGLE)
        b.ClearProp('_bond_type')
    # remove linear linkers
    atom = get_next_atom_to_remove(topo, ignore_cycles = ignore_cycles)
    while atom:
        topo = remove_atom(atom)
        atom = get_next_atom_to_remove(topo, ignore_cycles = ignore_cycles)
    # remove bubbles
    if not ignore_cycles:
        # bubbles
        atom = get_next_atom_to_remove(topo, bubble = True)
        while atom:
            topo = remove_atom(atom, bubble = True)
            atom = get_next_atom_to_remove(topo, bubble = True)
        # remaining linear linkers
        atom = get_next_atom_to_remove(topo)
        while atom:
            topo = remove_atom(atom)
            atom = get_next_atom_to_remove(topo)
    # sanitize
    _Chem.SanitizeMol(topo)
    
    return topo


