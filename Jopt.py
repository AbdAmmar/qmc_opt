
import sys, os
QMCCHEM_PATH=os.environ["QMCCHEM_PATH"]
sys.path.insert(0,QMCCHEM_PATH+"/EZFIO/Python/")
from ezfio import ezfio
from datetime import datetime
import time
import numpy as np
import subprocess
from modif_powell_imp import my_fmin_powell



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

# ---

def f(x):

    global i_fev
    global memo_energy

    print('\n eval {} of f on:'.format(i_fev))
    print(' nuc param Jast = {}'.format(x[:-1]))
    print(' b   param Jast = {}'.format(x[-1]))

    h = str(x)
    if h in memo_energy:
        return memo_energy[h]

    i_fev = i_fev + 1

    set_params_pen(x[:-1])
    set_params_b(x[-1])
    set_vmc_params(block_time_f, total_time_f)

    loc_err = 10.
    ii      = 1
    ii_max  = 5 
    energy  = None
    err     = None
    while( Eloc_err_th < loc_err ):

        run_qmc()
        energy, err = get_energy()
        var_en, _   = get_variance()

        if( (energy is None) or (err is None) ):
            continue

        elif( memo_energy['fmin'] < (energy-2.*err) ):
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

    memo_energy[h]      = energy + err
    memo_energy['fmin'] = min(energy, memo_energy['fmin'])

    return( energy + var_weight * var_en )

# ---







if __name__ == '__main__':

    t0 = time.time()

    EZFIO_file = sys.argv[1] 
    ezfio.set_file(EZFIO_file)

    print(" Today's date:", datetime.now() )
    print(" EZFIO file = {}".format(EZFIO_file))


    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
    #                     PARAMETERS

    block_time_f = 60
    total_time_f = 200

    Eloc_err_th = 0.01

    var_weight = 0.02

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    # map nuclei to a list
    atom_map = make_atom_map()

    n_par = len(atom_map)  # nb of nclear parameters
    n_par = n_par + 1      # e-e parameter b
    print(' nb of parameters = {}'.format(n_par))

    x = get_params_pen()
    print(' initial pen: {}'.format(x))
    b_par = get_params_b()
    print(' initial   b: {}'.format(b_par))
    x = np.append(x, b_par)


    sys.stdout.flush()

    i_fev = 1
    memo_energy = {'fmin': 100.}

    x_min = [ (0.001) for _ in range(n_par) ] 
    x_max = [ (9.999) for _ in range(n_par) ]

    opt   = my_fmin_powell( f, x, x_min, x_max
			     #, xtol        = 0.01
			     #, ftol        = 0.01 
		             , maxfev      = 100
			     , full_output = 1
                             , verbose     = 1 )

    print(" x = "+str(opt))
    print(' number of function evaluations = {}'.format(i_fev))
    print(' memo_energy: {}'.format(memo_energy))

    print(" end after {:.3f} minutes".format((time.time()-t0)/60.) )

    # !!!
# !!!


