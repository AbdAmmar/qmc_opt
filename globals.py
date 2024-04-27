
import sys

vmc_block_time = 60
vmc_total_time = 200

Eloc_err_th = 0.01
var_weight = 0.00

i_fev = 1
memo_energy = {'fmin': 100.}

list_j2e_type = ["None", "Mu", "Mu_Nu", "Mur", "Boys", "Boys_Handy", "Qmckl"]
list_env_type = ["None", "Prod_Gauss", "Sum_Gauss", "Sum_Slat", "Sum_Quartic"]
list_j1e_type = ["None", "Gauss", "Charge_Harmonizer", "Charge_Harmonizer_AO"]

j2e_type = list_j2e_type[5]
env_type = list_env_type[0]
j1e_type = list_j1e_type[0]

thresh_tcscf = 1.e-6
n_it_tcscf_max = 10

mu = 0.87

# TODO
list_H = [2, 3]
env_expo_H = 100000.0

do_scf = True
optimize_orb = False

EZFIO_file = sys.argv[1]

file_out="RESULTS_{}.out".format(EZFIO_file)


