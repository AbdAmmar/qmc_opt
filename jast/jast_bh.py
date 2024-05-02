
import numpy as np
import sys
import random
import globals
from utils.qp2_utils import clear_tcscf_orbitals, run_tcscf
from utils.qmcchem_utils import set_vmc_params, run_qmc, get_energy, get_variance, get_var_Htc
from utils.utils import append_to_output
from jast.jast_param import get_jbh_m, get_jbh_n, get_jbh_o, get_jbh_c, get_jbh_size, get_jbh_en, get_jbh_ee
from jast.jast_param import set_jbh_m, set_jbh_n, set_jbh_o, set_jbh_c, set_jbh_size, set_jbh_en, set_jbh_ee

# ---

def f_jbh(x, atom_map, ezfio):

    h = str(x)
    if h in globals.memo_res:
        return globals.memo_res[h]

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
            globals.memo_res[h]      = energy + err
            globals.memo_res['fmin'] = min(energy, globals.memo_res['fmin'])
            return energy + globals.var_weight * var_en

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

        elif(globals.memo_res['fmin'] < (energy-2.*err)):
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

    globals.memo_res[h]      = energy + err
    globals.memo_res['fmin'] = min(energy, globals.memo_res['fmin'])

    if(globals.optimize_orb):
        ezfio.set_mo_basis_mo_coef(mohf.T)

    return energy + globals.var_weight * var_en

# ---

def vartc_jbh_vmc(x, atom_map, ezfio):

    h = str(x)
    if h in globals.memo_res:
        return globals.memo_res[h]

    append_to_output('\n eval {} of f on:'.format(globals.i_fev))
    append_to_output(' x = ' + '  '.join([f"{xx:.7f}" for xx in x]))

    globals.i_fev = globals.i_fev + 1

    # UPDATE PARAMETERS
    map_x_to_jbh_coef(atom_map, ezfio, x)
    print_jbh(atom_map, ezfio)

    set_vmc_params()

    loc_err = 10.
    ii      = 1
    ii_max  = 5 
    energy  = None
    err     = None
    while(globals.Eloc_err_th < loc_err):

        run_qmc()
        energy, err = get_energy()
        var_en, _ = get_variance()
        var_Htc, var_Htc_err = get_var_Htc()

        if((energy is None) or (err is None)):
            continue

        elif(globals.memo_res['fmin'] < var_Htc):
            append_to_output(" %d energy: %f  %f %f"%(ii, energy, err, var_en))
            append_to_output(" var_Htc: %f %f"%(var_Htc, var_Htc_err))
            sys.stdout.flush()
            break

        else:
            loc_err = err
            append_to_output(" %d energy: %f  %f %f"%(ii, energy, err, var_en))
            append_to_output(" var_Htc: %f %f"%(var_Htc, var_Htc_err))
            sys.stdout.flush()
            if( ii_max < ii ):
                break
            ii += 1

    globals.memo_res[h]      = var_Htc
    globals.memo_res['fmin'] = min(var_Htc, globals.memo_res['fmin'])

    return var_Htc

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

