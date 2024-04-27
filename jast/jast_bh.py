
import numpy as np
import sys
import random
import globals
from utils.qp2_utils import clear_tcscf_orbitals, run_tcscf
from utils.qmcchem_utils import set_vmc_params, run_qmc, get_energy, get_variance
from utils.utils import append_to_output
from jast.jast_param import get_jbh_m, get_jbh_n, get_jbh_o, get_jbh_c, get_jbh_size, get_jbh_en, get_jbh_ee
from jast.jast_param import set_jbh_m, set_jbh_n, set_jbh_o, set_jbh_c, set_jbh_size, set_jbh_en, set_jbh_ee


def f_jbh(x, atom_map, ezfio):

    h = str(x)
    if h in globals.memo_energy:
        return globals.memo_energy[h]

    append_to_output('\n eval {} of f on:'.format(globals.i_fev))
    append_to_output(' x = ' + '  '.join([f"{xx:.7f}" for xx in x]))

    globals.i_fev = globals.i_fev + 1

    # UPDATE PARAMETERS
    map_x_to_jbh_coef(atom_map, ezfio, x)
    print_jbh(atom_map, ezfio)

    if(globals.optimize_orb):
        clear_tcscf_orbitals(ezfio)
        e_tcscf, c_tcscf = run_tcscf(ezfio)
        if c_tcscf:
            append_to_output(' tc-scf energy = {}'.format(e_tcscf))
            mohf = np.array(ezfio.get_mo_basis_mo_coef()).T
            mor  = np.array(ezfio.get_bi_ortho_mos_mo_r_coef()).T
            ezfio.set_mo_basis_mo_coef(mor.T)
        else:
            append_to_output(' tc-scf did not converged')
            energy = 100.0 + 10.0 * random.random()
            var_en = 100.0 + 10.0 * random.random()
            err    =         10.0 * random.random()
            append_to_output(" energy: %f  %f %f"%(energy, err, var_en))
            globals.memo_energy[h]      = energy + err
            globals.memo_energy['fmin'] = min(energy, globals.memo_energy['fmin'])
            return energy + globals.var_weight * var_en

    # GET VMC energy & variance 
    set_vmc_params()

    loc_err = 10.
    ii      = 1
    ii_max  = 5 
    energy  = None
    err     = None
    while(globals.Eloc_err_th < loc_err):

        run_qmc()
        energy, err = get_energy()
        var_en, _   = get_variance()

        if((energy is None) or (err is None)):
            continue

        elif(globals.memo_energy['fmin'] < (energy-2.*err)):
            append_to_output(" %d energy: %f  %f %f"%(ii, energy, err, var_en))
            sys.stdout.flush()
            break

        else:
            loc_err = err
            append_to_output(" %d energy: %f  %f %f"%(ii, energy, err, var_en))
            sys.stdout.flush()
            if( ii_max < ii ):
                break
            ii += 1

    globals.memo_energy[h]      = energy + err
    globals.memo_energy['fmin'] = min(energy, globals.memo_energy['fmin'])
    ezfio.set_mo_basis_mo_coef(mohf.T)

    return energy + globals.var_weight * var_en

# ---

def init_jbh(atom_map, ezfio):

    jbh_size = get_jbh_size(ezfio)

    jbh_m = get_jbh_m(jbh_size, atom_map, ezfio)
    jbh_n = get_jbh_n(jbh_size, atom_map, ezfio)
    jbh_o = get_jbh_o(jbh_size, atom_map, ezfio)

    exist = ezfio.has_jastrow_jbh_c()
    if(not exist):
        c_random = [random.uniform(-1, 1) for _ in range(len(atom_map)*jbh_size)]
        set_jbh_c(c_random, jbh_size, atom_map, ezfio)
    else:
        jbh_c = ezfio.get_jastrow_jbh_c()
    jbh_c = get_jbh_c(jbh_size, atom_map, ezfio)

    print_jbh(atom_map, ezfio)

    x = map_jbh_coef_to_x(atom_map, ezfio)

    xx_min, xx_max = -10.0, +10.0
    x_min = [xx_min for _ in x]
    x_max = [xx_max for _ in x]

    print(x)
    print(x_min)
    print(x_max)

    args = (jbh_size, jbh_m, jbh_n, jbh_o)

    return args, x, x_min, x_max

# ---

def map_jbh_coef_to_x(atom_map, ezfio):

    jbh_size = get_jbh_size(ezfio)

    jbh_m = get_jbh_m(jbh_size, atom_map, ezfio)
    jbh_n = get_jbh_n(jbh_size, atom_map, ezfio)
    jbh_c = get_jbh_c(jbh_size, atom_map, ezfio)

    x = []
    for i,a in enumerate(atom_map):
        j = a[0]
        for p in range(jbh_size):
            ii = i*jbh_size + p
            if(jbh_m[ii] == jbh_n[ii] == 0):
                if(j == 0):
                    if(p > 0):
                        x.append(jbh_c[ii]) 
            else:
                x.append(jbh_c[ii]) 

    return x

# ---

def map_x_to_jbh_coef(atom_map, ezfio, x):

    jbh_size = get_jbh_size(ezfio)

    jbh_m = get_jbh_m(jbh_size, atom_map, ezfio)
    jbh_n = get_jbh_n(jbh_size, atom_map, ezfio)

    jbh_c = [0.0 for _ in jbh_m]

    jj = 0
    for i,a in enumerate(atom_map):
        j = a[0]
        for p in range(jbh_size):
            ii = i*jbh_size + p
            if(jbh_m[ii] == jbh_n[ii] == 0):
                if(j == 0):
                    if(p == 0):
                        jbh_c[ii] = 0.5
                    else:
                        jbh_c[ii] = x[jj]
                        jj += 1
            else:
                jbh_c[ii] = x[jj]
                jj += 1

    set_jbh_c(jbh_c)

# ---

def print_jbh(atom_map, ezfio):

    jbh_size = get_jbh_size(ezfio)
    append_to_output(" jbh_size = {}".format(jbh_size))

    jbh_m = get_jbh_m(jbh_size, atom_map, ezfio)
    jbh_n = get_jbh_n(jbh_size, atom_map, ezfio)
    jbh_o = get_jbh_o(jbh_size, atom_map, ezfio)
    jbh_c = get_jbh_c(jbh_size, atom_map, ezfio)

    for ii in range(len(atom_map)):
        append_to_output(" ATOM: {}   m       n       o       c".format(ii+1))
        for jj in range(jbh_size):
            i = ii*jbh_size+jj
            m = jbh_m[i]
            n = jbh_n[i]
            o = jbh_o[i]
            c = jbh_c[i]
            append_to_output("        {:4d}    {:4d}    {:4d}    {:+3.5f}".format(m, n, o, c))
        append_to_output("")

    sys.stdout.flush()

# ---

def init_jbh9_water(atom_map, ezfio):

    jbh_size = 9

    jbh_m = [[0, 0, 0, 0, 2, 3, 4, 2, 2], [0, 0, 0, 0, 2, 3, 4, 2, 2], [0, 0, 0, 0, 2, 3, 4, 2, 2]]
    jbh_n = [[0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0]]
    jbh_o = [[1, 2, 3, 4, 0, 0, 0, 0, 2], [1, 2, 3, 4, 0, 0, 0, 0, 2], [1, 2, 3, 4, 0, 0, 0, 0, 2]]
    jbh_c = [[+0.5, -0.57070, +0.49861, -0.78663, +0.01990, +0.13386, -0.60446, -1.67160, +1.36590],
             [+0.0, +0.0    , +0.0    , +0.0    , +0.12063, -0.18527, +0.12324, -0.11187, -0.06558],
             [+0.0, +0.0    , +0.0    , +0.0    , +0.12063, -0.18527, +0.12324, -0.11187, -0.06558]]

    jbh_ee = [1.0, 1.0, 1.0]
    jbh_en = [1.0, 1.0, 1.0]

    ezfio.set_jastrow_jbh_size(jbh_size)

    ezfio.set_jastrow_jbh_o(jbh_o)
    ezfio.set_jastrow_jbh_m(jbh_m)
    ezfio.set_jastrow_jbh_n(jbh_n)
    ezfio.set_jastrow_jbh_c(jbh_c)

    ezfio.set_jastrow_jbh_ee(jbh_ee)
    ezfio.set_jastrow_jbh_en(jbh_en)

# ---
