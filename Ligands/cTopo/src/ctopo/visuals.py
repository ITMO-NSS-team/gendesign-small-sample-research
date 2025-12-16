'''Visualizing functionality for topos and skeletons'''

#%% Functions

def draw_skeletopo(mol) -> None:
    ''' '''
    for a in mol.GetAtoms():
        if a.GetProp('_ligand_atom_type') == 'DA':
            if a.GetSymbol() == '*':
                a.SetProp('_displayLabel', 'DA')
        elif a.GetProp('_ligand_atom_type') == 'SKL':
            if a.GetSymbol() == '*':
                a.SetProp('_displayLabel', '')
    
    return


