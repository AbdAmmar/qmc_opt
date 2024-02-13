
import sys, os
QMCCHEM_PATH=os.environ["QMCCHEM_PATH"]
sys.path.insert(0,QMCCHEM_PATH+"/EZFIO/Python/")
from ezfio import ezfio
from datetime import datetime
import time
import numpy as np

from modif_powell_imp import my_fmin_powell
from utils import make_atom_map, Hatom_map, f_envSumGauss_j1eGauss, run_scf, clear_tcscf_orbitals
from jast_param import get_env_coef, get_env_expo, get_j1e_size, get_j1e_coef, get_j1e_expo, get_mu
from jast_param import set_env_coef, set_env_expo, set_j1e_size, set_j1e_coef, set_j1e_expo, set_mu
import globals


if __name__ == '__main__':

    t0 = time.time()

    EZFIO_file = sys.argv[1] 
    ezfio.set_file(EZFIO_file)

    print(" Today's date:", datetime.now() )
    print(" EZFIO file = {}".format(EZFIO_file))

    E_scf = run_scf(ezfio, EZFIO_file)
    print(" HF energy = {}".format(E_scf))

    # JASTROW PARAMETRS
    ezfio.set_jastrow_j2e_type(globals.j2e_type)
    ezfio.set_jastrow_env_type(globals.env_type)
    ezfio.set_jastrow_j1e_type(globals.j1e_type)
    set_mu(globals.mu, ezfio)
    ezfio.set_tc_keywords_thresh_tcscf(globals.thresh_tcscf)
    ezfio.set_tc_keywords_n_it_tcscf_max(globals.n_it_tcscf_max)

    print(" j2e_type = {}".format(globals.j2e_type))
    print(" env_type = {}".format(globals.env_type))
    print(" j1e_type = {}".format(globals.j1e_type))
    print(" mu       = {}".format(globals.mu))

    # map nuclei to a list
    atom_map = make_atom_map(ezfio)
    #print(" atom_map: {}".format(atom_map))

    H_ind, H_nb, H_map = Hatom_map(ezfio)
    #print(" H_map: {}".format(H_map))
    #print(" H_ind: {}".format(H_ind))
    #print(" H_nb : {}".format(H_nb))

    n_nuc = len(atom_map)  # nb of nuclei withou repitition
    #print(' nb of unique nuclei = {}'.format(n_nuc))

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
    print(' total nb of parameters = {}'.format(n_par))

    x     = [(0.5) for _ in range(n_par_env)] + [(1.0) for _ in range(n_par_j1e_expo)] + [(-0.1) for _ in range(n_par_j1e_coef)] 
    x_min = [(0.1) for _ in range(n_par_env)] + [(0.1) for _ in range(n_par_j1e_expo)] + [(-9.9) for _ in range(n_par_j1e_coef)] 
    x_max = [(4.9) for _ in range(n_par_env)] + [(9.9) for _ in range(n_par_j1e_expo)] + [(-0.1) for _ in range(n_par_j1e_coef)]

    print(' starting point: {}'.format(x))
    print(' parameters are bounded between:')
    print(' x_min: {}'.format(x_min))
    print(' x_max: {}'.format(x_max))

    sys.stdout.flush()

    args = ( n_nuc, atom_map
           , H_ind, n_par_env
           , n_par_j1e_expo, j1e_size
           , ezfio, EZFIO_file )

    opt = my_fmin_powell( f_envSumGauss_j1eGauss, x, x_min, x_max
                        , args        = args
        		, xtol        = 0.1
        		, ftol        = 0.1
        	        , maxfev      = 100
        		, full_output = 1
                        , verbose     = 1 )

    print(" x = "+str(opt))
    print(' number of function evaluations = {}'.format(globals.i_fev))
    print(' memo_energy: {}'.formatglo(globals.memo_energy))

    print(" end after {:.3f} minutes".format((time.time()-t0)/60.) )




