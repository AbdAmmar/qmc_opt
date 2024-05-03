
import sys
from ezfio import ezfio

# ---

EZFIO_file = sys.argv[1]
ezfio.set_file(EZFIO_file)

# ---

vmc_block_time = 30
vmc_total_time = 150

Eloc_err_th = 0.01
var_weight = 0.00

i_fev = 1
memo_res = {'fmin': 1e10}

list_j2e_type = ["None", "Mu", "Mu_Nu", "Mur", "Boys", "Boys_Handy", "Qmckl"]
list_env_type = ["None", "Prod_Gauss", "Sum_Gauss", "Sum_Slat", "Sum_Quartic"]
list_j1e_type = ["None", "Gauss", "Charge_Harmonizer", "Charge_Harmonizer_AO"]

j2e_type = list_j2e_type[0]
env_type = list_env_type[0]
j1e_type = list_j1e_type[0]

thresh_tcscf = 1.e-6
n_it_tcscf_max = 10

mu = 0.87

env_expo_H = 100000.0

do_scf = False
optimize_orb = False


file_out="RESULTS_{}.out".format(EZFIO_file)


