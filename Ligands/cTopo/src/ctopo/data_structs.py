'''Main classes and data structures'''

#%% Imports

import copy as _copy

from rdkit import Chem as _Chem

import ctopo.topology as _topo
import ctopo.representations as _repr
import ctopo.visuals as _vis


#%% Classes

class Ligand:
    ''' '''
    
    def __init__(self, smiles):
        self._mol = _Chem.MolFromSmiles(smiles)
        self.ligand = _copy.deepcopy(self._mol)
        _topo.annotate_ligand(self.ligand)
        self.skeleton = _topo.get_ligand_skeleton(self.ligand)
        self.topology = _topo.get_ligand_topology(self.ligand)
    
    
    def get_topology(self, atom_types_mapping = True, DA = True, da_skl_charges = False):
        ''' '''
        mol = _copy.deepcopy(self.topology)
        # atomic types via isotopes
        if atom_types_mapping:
            mol = _repr.set_atom_types_via_isotopes(mol)
        # explicit symbols for DAs
        if DA:
            mol = _repr.set_explicit_DAs(mol)
        if da_skl_charges:
            mol = _repr.restore_da_skl_charges(mol)
        # visual settings
        _vis.draw_skeletopo(mol)
        
        return mol
    
    
    def get_skeleton(self, atom_types_mapping = True, DA = True,
                     skeleton = False, da_skl_charges = False,
                     restore_skeleton_bonds = False):
        ''' '''
        mol = _copy.deepcopy(self.skeleton)
        # atomic types via isotopes
        if atom_types_mapping:
            mol = _repr.set_atom_types_via_isotopes(mol)
        # restore bonds
        if restore_skeleton_bonds:
            mol = _repr.restore_skeleton_bonds(mol)
        # explicit symbols for DAs
        if DA:
            mol = _repr.set_explicit_DAs(mol)
        if skeleton:
            mol = _repr.set_explicit_skeletons(mol)
        if da_skl_charges:
            mol = _repr.restore_da_skl_charges(mol)
        # visual settings
        _vis.draw_skeletopo(mol)
        
        return mol
    
    
    def get_complexophore(self, atom_types_mapping = True,
                          DA = True, skeleton = False, substituents = True,
                          da_skl_charges = False,
                          restore_skeleton_bonds = False,
                          restore_substituent_bonds = False):
        ''' '''
        mol = _copy.deepcopy(self.ligand)
        _topo.simplify_molecular_graph(mol)
        # atomic types via isotopes
        if atom_types_mapping:
            mol = _repr.set_atom_types_via_isotopes(mol)
        # restore bonds
        if restore_skeleton_bonds:
            mol = _repr.restore_skeleton_bonds(mol)
        if restore_substituent_bonds:
            mol = _repr.restore_substituent_bonds(mol)
        # explicit symbols for DAs
        if DA:
            mol = _repr.set_explicit_DAs(mol)
        if skeleton:
            mol = _repr.set_explicit_skeletons(mol)
        if substituents:
            mol = _repr.set_explicit_substituents(mol)
        if da_skl_charges:
            mol = _repr.restore_da_skl_charges(mol)
        # visual settings
        _vis.draw_skeletopo(mol)
        
        return mol


