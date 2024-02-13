
import numpy as np
import subprocess
import os, sys
import random
import globals
from globals import block_time_f, total_time_f, Eloc_err_th, var_weight
from jast_param import set_env_expo, set_j1e_expo, set_j1e_coef


# ---

def run_qmc(EZFIO_file):
    return subprocess.check_output(['qmcchem', 'run', EZFIO_file])

def stop_qmc(EZFIO_file):
    subprocess.check_output(['qmcchem', 'stop', EZFIO_file])

def set_vmc_params(block_time, total_time, EZFIO_file):
    subprocess.check_output([ 'qmcchem', 'edit', '-c'
                            , '-j', globals.j2e_type
                            , '-t', str(total_time)
                            , '-l', str(block_time)
                            , EZFIO_file] )

def get_energy(EZFIO_file):
    buffer = subprocess.check_output(
            ['qmcchem', 'result', '-e', 'e_loc', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, energy, error = [float(x) for x in buffer.split()]
        return energy, error
    else:
        return None, None

def get_variance(EZFIO_file):
    buffer = subprocess.check_output(['qmcchem', 'result', '-e', 'e_loc_qmcvar', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, variance, error = [float(x) for x in buffer.split()]
        return variance, error
    else:
        return None, None

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

def clear_tcscf_orbitals(EZFIO_file):
    mor = os.path.join(EZFIO_file, "bi_ortho_mos", "mo_r_coef.gz")
    mol = os.path.join(EZFIO_file, "bi_ortho_mos", "mo_l_coef.gz")
    if os.path.exists(mor):
        subprocess.check_call(['rm', mor])
    if os.path.exists(mol):
        subprocess.check_call(['rm', mol])

def run_tcscf(ezfio, EZFIO_file):
    with open("tc_scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'tc_scf', EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    e_tcscf = ezfio.get_tc_scf_bitc_energy()
    c_tcscf = ezfio.get_tc_scf_converged_tcscf()
    return e_tcscf, c_tcscf

# ---

def f_envSumGauss_j1eGauss(x, args):

    n_nuc, atom_map, H_ind, H_nb, n_par_env, n_par_j1e_expo, j1e_size, ezfio, EZFIO_file = args

    print('\n eval {} of f on:'.format(globals.i_fev))
    print('\n x = {}'.format(x))

    env_expo = []
    jj = 0
    for ii in range(n_nuc):
        if(ii == H_ind):
            env_expo.append(globals.env_expo_H)
        else:
            env_expo.append(x[jj])
            jj += 1

    j1e_expo = x[n_par_env:n_par_env+n_par_j1e_expo]
    j1e_coef = x[n_par_env+n_par_j1e_expo:]

    print(' env expo: {}'.format(env_expo))
    print(' j1e expo: {}'.format(j1e_expo))
    print(' j1e coef: {}'.format(j1e_expo))
    sys.stdout.flush()

    h = str(x)
    if h in globals.memo_energy:
        return globals.memo_energy[h]

    globals.i_fev = globals.i_fev + 1

    # UPDATE PARAMETERS
    set_env_expo(env_expo, atom_map, ezfio)
    set_j1e_expo(j1e_expo, j1e_size, atom_map, ezfio)
    set_j1e_coef(j1e_coef, j1e_size, atom_map, ezfio)

    # OPTIMIZE ORBITALS
    clear_tcscf_orbitals(EZFIO_file)
    e_tcscf, c_tcscf = run_tcscf(ezfio, EZFIO_file)
    if c_tcscf:
        print(' tc-scf converged')
        print(' tc-scf energy = {}'.format(e_tcscf))
        mohf = np.array(ezfio.get_mo_basis_mo_coef()).T
        mor  = np.array(ezfio.get_bi_ortho_mos_mo_r_coef()).T
        ezfio.set_mo_basis_mo_coef(mor.T)
    else:
        print(' tc-scf did not converged')
        return 100.0 + 10.0 * random.random()

    # GET VMC energy & variance 
    set_vmc_params(block_time_f, total_time_f, EZFIO_file)

    loc_err = 10.
    ii      = 1
    ii_max  = 5 
    energy  = None
    err     = None
    while(Eloc_err_th < loc_err):

        run_qmc(EZFIO_file)
        energy, err = get_energy(EZFIO_file)
        var_en, _   = get_variance(EZFIO_file)

        if((energy is None) or (err is None)):
            continue

        elif(globals.memo_energy['fmin'] < (energy-2.*err)):
            print(" %d energy: %f  %f %f"%(ii, energy, err, var_en))
            sys.stdout.flush()
            break

        else:
            loc_err = err
            print(" %d energy: %f  %f %f"%(ii, energy, err, var_en))
            sys.stdout.flush()
            if( ii_max < ii ):
                break
            ii += 1

    globals.memo_energy[h]      = energy + err
    globals.memo_energy['fmin'] = min(energy, globals.memo_energy['fmin'])
    ezfio.set_mo_basis_mo_coef(mohf.T)

    return energy + var_weight * var_en

# ---

