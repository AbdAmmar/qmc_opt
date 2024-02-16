
import sys, os
QMCCHEM_PATH=os.environ["QMCCHEM_PATH"]
sys.path.insert(0,QMCCHEM_PATH+"/EZFIO/Python/")
from ezfio import ezfio
from datetime import datetime
import time
import numpy as np
from opt.scipy_powell import fmin_powell
from opt.modif_powell_imp import my_fmin_powell
from utils.qp2_utils import make_atom_map, Hatom_map, run_scf
from utils.utils import append_to_output
from utils.rosen import f_rosen, init_rosen
from jast.jast_mu_env_gauss import f_envSumGauss_j1eGauss, init_envSumGauss_j1eGauss
from jast.jast_param import set_mu
import globals


if __name__ == '__main__':

    t0 = time.time()

    ezfio.set_file(globals.EZFIO_file)

    append_to_output(" Today's date: {}".format(datetime.now()))
    append_to_output(" EZFIO file = {}".format(globals.EZFIO_file))

    if(globals.do_scf):
        E_scf = run_scf(ezfio)
        append_to_output(" HF energy = {}".format(E_scf))

    # JASTROW PARAMETRS
    ezfio.set_jastrow_j2e_type(globals.j2e_type)
    ezfio.set_jastrow_env_type(globals.env_type)
    ezfio.set_jastrow_j1e_type(globals.j1e_type)
    set_mu(globals.mu, ezfio)
    ezfio.set_tc_keywords_thresh_tcscf(globals.thresh_tcscf)
    ezfio.set_tc_keywords_n_it_tcscf_max(globals.n_it_tcscf_max)

    append_to_output(" j2e_type = {}".format(globals.j2e_type))
    append_to_output(" env_type = {}".format(globals.env_type))
    append_to_output(" j1e_type = {}".format(globals.j1e_type))
    append_to_output(" mu       = {}".format(globals.mu))

    # map nuclei to a list
    atom_map = make_atom_map(ezfio)
    #print(" atom_map: {}".format(atom_map))

    H_ind, H_nb, H_map = Hatom_map(ezfio)
    #print(" H_map: {}".format(H_map))
    #print(" H_ind: {}".format(H_ind))
    #print(" H_nb : {}".format(H_nb))

    n_nuc = len(atom_map)  # nb of nuclei withou repitition
    #print(' nb of unique nuclei = {}'.format(n_nuc))

    n_par_env, j1e_size, n_par_j1e_expo, n_par_j1e_coef, n_par_j1e, n_par, x, x_min, x_max = init_envSumGauss_j1eGauss(n_nuc, H_nb, ezfio)
    args = ( n_nuc, atom_map
           , H_ind, n_par_env
           , n_par_j1e_coef, j1e_size
           , ezfio )


    #n_par = 5
    #x, x_min, x_max = init_rosen(n_par)
    #args = ()

    append_to_output(' total nb of parameters = {}'.format(n_par))
    append_to_output(' starting point: {}'.format(x))
    append_to_output(' parameters are bounded between:')
    append_to_output(' x_min: {}'.format(x_min))
    append_to_output(' x_max: {}'.format(x_max))

    sys.stdout.flush()


#    opt = my_fmin_powell( 
#                          #f_envSumGauss_j1eGauss
#                          f_rosen
#                        , x, x_min, x_max
#                        , args        = args
#                        , xtol        = 0.1
#                        , ftol        = 0.1
#                        , maxfev      = 100
#                        , maxiter     = 5
#                        , full_output = 1
#                        , verbose     = 1 )

    bounds = {
        "lb": np.array(x_min),
        "ub": np.array(x_max)
    }
    opt = fmin_powell( 
                       #f_rosen
                       f_envSumGauss_j1eGauss
                     , x
                     , args        = args
                     , bounds      = bounds
                     , xtol        = 0.1
                     , ftol        = 0.1
        	      , maxfev      = 100
        	      , maxiter     = 5
                     , full_output = 1 )


    append_to_output(" x = "+str(opt))
    append_to_output(' number of function evaluations = {}'.format(globals.i_fev))
    append_to_output(' memo_energy: {}'.format(globals.memo_energy))

    append_to_output(" end after {:.3f} minutes".format((time.time()-t0)/60.) )




