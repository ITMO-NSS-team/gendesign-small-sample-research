'''Graph functionality via NetworkX'''

#%% Imports

import networkx as _nx

from rdkit import Chem as _Chem

import typing as _tp


#%% Functions

def mol_to_graph(mol: _Chem.rdchem.Mol) -> _nx.classes.graph.Graph:
    '''Converts RDKit Mol object to NetworkX graph
    
    Arguments:
        - mol (Chem.rdchem.Mol): any RDKit Molecule object
    
    Returns:
        nx.classes.graph.Graph: NetworkX Graph object containing connectivity only
    '''
    G = _nx.Graph()
    # add atoms
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    # add atoms
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    
    return G


def get_all_shortest_paths(mol: _Chem.rdchem.Mol, i: int, j: int) -> _tp.List[_tp.List[int]]:
    '''Returns all shortest path between two given atoms
    
    Arguments:
        - mol (Chem.rdchem.Mol): any RDKit Molecule object
        - i (int): source
        - j (int): sink
    
    Returns:
        List[List[int]]: List of shortest paths
    '''
    G = mol_to_graph(mol)
    paths = list(_nx.all_shortest_paths(G, i, j))
    
    return paths


