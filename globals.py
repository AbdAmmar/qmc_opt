
import sys

vmc_block_time = 60
vmc_total_time = 200

Eloc_err_th = 0.01
var_weight = 0.00

i_fev = 1
memo_energy = {'fmin': 100.}

j2e_type = "Mu"
env_type = "Sum_Gauss"
j1e_type = "Gauss"
thresh_tcscf = 1.e-6
n_it_tcscf_max = 10
mu = 0.87

list_H = [2, 3]
env_expo_H = 100000.0

do_scf = True

EZFIO_file = sys.argv[1]

file_out="RESULTS_{}.out".format(EZFIO_file)
