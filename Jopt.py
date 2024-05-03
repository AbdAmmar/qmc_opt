
import sys, os
QMCCHEM_PATH=os.environ["QMCCHEM_PATH"]
sys.path.insert(0,QMCCHEM_PATH+"/EZFIO/Python/")
from ezfio import ezfio
from datetime import datetime
import time
import numpy as np
from opt.scipy_powell import fmin_powell

from utils.qp2_utils import run_scf
from utils.utils import append_to_output

from jast.jast_mu_env_gauss import f_envSumGauss_j1eGauss
from jast.jast_bh import f_jbh, init_jbh, vartc_jbh_vmc

from globals import ezfio, EZFIO_file
from globals import do_scf
from globals import optimize_orb, thresh_tcscf, n_it_tcscf_max
from globals import j2e_type, env_type, j1e_type

# ---



if __name__ == '__main__':

    t0 = time.time()

    append_to_output(" Today's date: {}\n".format(datetime.now()))
    append_to_output(" EZFIO file = {}\n".format(EZFIO_file))

    if(do_scf):
        E_scf = run_scf(ezfio)
        append_to_output(" HF energy = {}\n".format(E_scf))
    if(optimize_orb):
        ezfio.set_tc_keywords_thresh_tcscf(thresh_tcscf)
        ezfio.set_tc_keywords_n_it_tcscf_max(n_it_tcscf_max)

    # JASTROW PARAMETRS
    ezfio.set_jastrow_j2e_type(j2e_type)
    ezfio.set_jastrow_env_type(env_type)
    ezfio.set_jastrow_j1e_type(j1e_type)

    append_to_output(" j2e_type = {}\n".format(j2e_type))
    append_to_output(" env_type = {}\n".format(env_type))
    append_to_output(" j1e_type = {}\n".format(j1e_type))

    x, x_min, x_max = init_jbh()
    args = ()

    bounds = {
        "lb": np.array(x_min),
        "ub": np.array(x_max)
    }


    opt = fmin_powell( 
                       #f_envSumGauss_j1eGauss
                       vartc_jbh_vmc
                     , x
                     , args        = args
                     , bounds      = bounds
                     , xtol        = 0.01
                     , ftol        = 0.01
                     , maxfev      = 1000
                     , maxiter     = 1000
                     , full_output = 1 )


    append_to_output(" x = "+str(opt)+"\n")
    append_to_output(' number of function evaluations = {}\n'.format(globals.i_fev))
    append_to_output(' memo_res: {}\n'.format(globals.memo_res))

    append_to_output(" end after {:.3f} minutes\n".format((time.time()-t0)/60.) )



