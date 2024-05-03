
from globals import ezfio

from utils.utils import append_to_output

# ---

def make_atom_map():
    labels = {}
    rank   = 0
    for i,k in enumerate(ezfio.nuclei_nucl_label):
        if k in labels:
            labels[k].append(i)
        else:
            labels[k] = [rank, i]
            rank += 1
    atom_map = [[] for i in range(rank)]
    for atom in labels.keys():
        l = labels[atom]
        atom_map[l[0]] = l[1:]
    return atom_map

atom_map = make_atom_map()

N_atom = 0
unique_atom_list = []
all_atoms = ezfio.get_nuclei_nucl_label()
for atoms in atom_map:
    unique_atom_list.append(all_atoms[atoms[0]])
    N_atom += len(atoms)
N_unique_atom = len(unique_atom_list)

append_to_output(" nb of atoms = {}\n".format(N_atom))
append_to_output(" nb of unique atoms = {}\n".format(N_unique_atom))

# ---

def Hatom_map():
    labels = {}
    rank   = 0
    for i,k in enumerate(ezfio.nuclei_nucl_label):
        if k in labels:
            labels[k].append(i)
        else:
            labels[k] = [rank, i]
            rank += 1
    list_H = []
    H_ind = -1
    for atom in labels.keys():
        if(atom == "H"):
            l = labels[atom]
            H_ind = l[0]
            list_H = l[1:]
            break
    H_nb = len(list_H)
    return H_ind, H_nb, list_H

# ---


