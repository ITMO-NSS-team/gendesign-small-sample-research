'''Functions to change molecule's representation'''

#%% Imports

import copy as _copy

from rdkit import Chem as _Chem


#%% Functions

def set_atom_types_via_isotopes(mol):
    '''Returns copy of input molecule with atomic types encoded via isotopes'''
    mol = _copy.deepcopy(mol)
    props = {'DA': 2, 'SKL': 1}
    for a in mol.GetAtoms():
        prop = props.get(a.GetProp('_ligand_atom_type'), 0)
        a.SetIsotope(prop)
    
    return mol


def set_explicit_DAs(mol):
    '''Returns copy of input molecule with DAs set explicitly'''
    mol = _copy.deepcopy(mol)
    for a in mol.GetAtoms():
        if a.GetProp('_ligand_atom_type') == 'DA':
            asym = a.GetProp('_atomic_symbol')
            anum = _Chem.Atom(asym).GetAtomicNum()
            a.SetAtomicNum(anum)
            a.SetProp('_displayLabel', asym)
            a.SetNumRadicalElectrons(0)
    
    return mol


def set_explicit_skeletons(mol):
    '''Returns copy of input molecule with skeleton atoms set explicitly'''
    mol = _copy.deepcopy(mol)
    for a in mol.GetAtoms():
        if a.GetProp('_ligand_atom_type') == 'SKL':
            asym = a.GetProp('_atomic_symbol')
            anum = _Chem.Atom(asym).GetAtomicNum()
            a.SetAtomicNum(anum)
            a.SetProp('_displayLabel', asym)
            a.SetNumRadicalElectrons(0)
    
    return mol


def set_explicit_substituents(mol):
    '''Returns copy of input molecule with substituent atoms set explicitly'''
    mol = _copy.deepcopy(mol)
    for a in mol.GetAtoms():
        if a.GetProp('_ligand_atom_type') == 'SUB':
            asym = a.GetProp('_atomic_symbol')
            anum = _Chem.Atom(asym).GetAtomicNum()
            a.SetAtomicNum(anum)
            a.ClearProp('_displayLabel')
            a.SetNumExplicitHs(a.GetIntProp('_number_of_hydrogens'))
            a.SetFormalCharge(a.GetIntProp('_atomic_charge'))
            a.SetNumRadicalElectrons(0)
    
    return mol


def restore_da_skl_charges(mol):
    '''Returns copy of input molecule with atomic charges at DAs and SKLs restored'''
    mol = _copy.deepcopy(mol)
    for a in mol.GetAtoms():
        if a.GetProp('_ligand_atom_type') not in ('DA', 'SKL'):
            continue
        if a.GetIntProp('_atomic_charge') != 0:
            charge = a.GetIntProp('_atomic_charge')
            a.SetFormalCharge(charge)
    
    return mol


def get_skeleton_bonds(mol):
    '''Returns list of indexes of skeleton bonds'''
    skeleton_bonds = []
    for b in mol.GetBonds():
        type1 = b.GetBeginAtom().GetProp('_ligand_atom_type')
        type2 = b.GetEndAtom().GetProp('_ligand_atom_type')
        if type1 == 'SUB' or type2 == 'SUB':
            continue
        skeleton_bonds.append(b.GetIdx())
    
    return skeleton_bonds


def restore_skeleton_bonds(mol):
    '''Returns copy of input molecule with restored skeleton bond types'''
    mol = _copy.deepcopy(mol)
    skeleton_bonds = get_skeleton_bonds(mol)
    for b in mol.GetBonds():
        if b.GetIdx() not in skeleton_bonds:
            continue
        bond_type = b.GetProp('_reduced_bond_type')
        bond_type = getattr(_Chem.BondType, bond_type)
        b.SetBondType(bond_type)
    
    return mol


def restore_substituent_bonds(mol):
    '''Returns copy of input molecule with restored substituent bond types'''
    mol = _copy.deepcopy(mol)
    skeleton_bonds = get_skeleton_bonds(mol)
    for b in mol.GetBonds():
        if b.GetIdx() in skeleton_bonds:
            continue
        bond_type = b.GetProp('_reduced_bond_type')
        bond_type = getattr(_Chem.BondType, bond_type)
        b.SetBondType(bond_type)
    
    return mol


