
import numpy as np
import sys
import random
import globals
from qp2_utils import clear_tcscf_orbitals, run_tcscf
from qmcchem_utils import set_vmc_params, run_qmc, get_energy, get_variance
from jast_param import set_env_expo, set_j1e_expo, set_j1e_coef, get_j1e_size


#def f_envSumGauss_j1eGauss(x, args):
#    n_nuc, atom_map, H_ind, n_par_env, n_par_j1e_expo, j1e_size, ezfio, EZFIO_file = args
def f_envSumGauss_j1eGauss(x, n_nuc, atom_map, H_ind, n_par_env, n_par_j1e_expo, j1e_size, ezfio, EZFIO_file):

    h = str(x)
    if h in globals.memo_energy:
        return globals.memo_energy[h]

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
    print(' j1e coef: {}'.format(j1e_coef))
    sys.stdout.flush()

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
        energy = 100.0 + 10.0 * random.random()
        var_en = 100.0 + 10.0 * random.random()
        err    =         10.0 * random.random()
        globals.memo_energy[h]      = energy + err
        globals.memo_energy['fmin'] = min(energy, globals.memo_energy['fmin'])
        return energy + globals.var_weight * var_en

    # GET VMC energy & variance 
    set_vmc_params(globals.block_time_f, globals.total_time_f, EZFIO_file)

    loc_err = 10.
    ii      = 1
    ii_max  = 5 
    energy  = None
    err     = None
    while(globals.Eloc_err_th < loc_err):

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

    return energy + globals.var_weight * var_en

# ---

def init_envSumGauss_j1eGauss(n_nuc, H_nb, ezfio):

    n_par_env = n_nuc
    if(H_nb != 0):
        n_par_env = n_par_env - 1
    #print(' nb of parameters for env = {}'.format(n_par_env))

    j1e_size = get_j1e_size(ezfio)
    #print(" j1e_size = {}".format(j1e_size))

    n_par_j1e_expo = j1e_size * n_nuc
    n_par_j1e_coef = j1e_size * n_nuc
    n_par_j1e = n_par_j1e_expo + n_par_j1e_coef
    #print(' nb of parameters for j1e = {}'.format(n_par_j1e))

    n_par = n_par_env + n_par_j1e
    #print(' total nb of parameters = {}'.format(n_par))

    x     = [(0.5) for _ in range(n_par_env)] + [(1.0) for _ in range(n_par_j1e_expo)] + [(-0.1) for _ in range(n_par_j1e_coef)]
    x_min = [(0.1) for _ in range(n_par_env)] + [(0.1) for _ in range(n_par_j1e_expo)] + [(-9.9) for _ in range(n_par_j1e_coef)]
    x_max = [(4.9) for _ in range(n_par_env)] + [(9.9) for _ in range(n_par_j1e_expo)] + [(-0.1) for _ in range(n_par_j1e_coef)]

    return n_par_env, j1e_size, n_par_j1e_expo, n_par_j1e_coef, n_par_j1e, n_par, x, x_min, x_max

# ---

