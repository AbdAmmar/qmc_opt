
import subprocess
import os


# ---

def make_atom_map(ezfio):
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

def Hatom_map(ezfio):
    labels = {}
    rank   = 0
    for i,k in enumerate(ezfio.nuclei_nucl_label):
        if k in labels:                
            labels[k].append(i)            
        else:
            labels[k] = [rank, i]
            rank += 1
    H_map = []
    H_ind = -1
    for atom in labels.keys():
        if(atom == "H"):
            l = labels[atom]
            H_ind = l[0]
            H_map = l[1:]
            break
    H_nb = len(H_map)
    return H_ind, H_nb, H_map 

# ---

def clear_tcscf_orbitals(ezfio, EZFIO_file):
    mor = os.path.join(EZFIO_file, "bi_ortho_mos", "mo_r_coef.gz")
    mol = os.path.join(EZFIO_file, "bi_ortho_mos", "mo_l_coef.gz")
    etc = os.path.join(EZFIO_file, "tc_scf", "bitc_energy")
    if os.path.exists(mor):
        subprocess.check_call(['rm', mor])
    if os.path.exists(mol):
        subprocess.check_call(['rm', mol])
    if os.path.exists(etc):
        subprocess.check_call(['rm', etc])
    ezfio.set_tc_scf_converged_tcscf(False)

def run_tcscf(ezfio, EZFIO_file):
    with open("tc_scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'tc_scf', EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    etc = os.path.join(EZFIO_file, "tc_scf", "bitc_energy")
    ctc = os.path.join(EZFIO_file, "tc_scf", "converged_tcscf")
    e_tcscf = None
    c_tcscf = False
    if os.path.exists(etc):
        e_tcscf = ezfio.get_tc_scf_bitc_energy()
    if os.path.exists(ctc):
        c_tcscf = ezfio.get_tc_scf_converged_tcscf()
    return e_tcscf, c_tcscf

# ---

def run_scf(ezfio, EZFIO_file):
    with open("scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'scf', EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    e_scf = ezfio.get_hartree_fock_energy()
    return e_scf

# ---

