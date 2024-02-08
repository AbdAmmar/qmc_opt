
import subprocess
import random
import globals
from globals import block_time_f, total_time_f, Eloc_err_th, var_weight
from jast_param import set_env_expo, set_j1e_expo, set_j1e_coef


# ---

def run_qmc(EZFIO_file):
    return subprocess.check_output(['qmcchem', 'run', EZFIO_file])

def stop_qmc(EZFIO_file):
    subprocess.check_output(['qmcchem', 'stop', EZFIO_file])

def set_vmc_params(block_time, total_time, EZFIO_file):
    subprocess.check_output([ 'qmcchem', 'edit', '-c'
                            , '-j', globals.j2e_type
                            , '-t', str(total_time)
                            , '-l', str(block_time)
                            , EZFIO_file] )

def get_energy(EZFIO_file):
    buffer = subprocess.check_output(
            ['qmcchem', 'result', '-e', 'e_loc', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, energy, error = [float(x) for x in buffer.split()]
        return energy, error
    else:
        return None, None

def get_variance(EZFIO_file):
    buffer = subprocess.check_output(['qmcchem', 'result', '-e', 'e_loc_qmcvar', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, variance, error = [float(x) for x in buffer.split()]
        return variance, error
    else:
        return None, None

# ---

def make_atom_map(ezfio):
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

def clear_tcscf_orbitals(EZFIO_file):
    mor = EZFIO_file + "/bi_ortho_mos/mo_r_coef.gz"
    mol = EZFIO_file + "/bi_ortho_mos/mo_l_coef.gz"
    subprocess.check_call(['rm', '{}'.format(mor)])
    subprocess.check_call(['rm', '{}'.format(mol)])

def run_tcscf(ezfio, EZFIO_file):
    with open("tc_scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'tc_scf', EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    e_tcscf = ezfio.get_tc_scf_bitc_energy()
    c_tcscf = ezfio.get_tc_scf_converged_tcscf()
    return e_tcscf, c_tcscf

# ---

def f_envSumGauss_j1eGauss(x, n_nuc, atom_map, j1e_size, ezfio, EZFIO_file):

    print('\n eval {} of f on:'.format(globals.i_fev))

    env_expo = x[:n_nuc]
    j1e_expo = x[n_nuc:2*n_nuc]
    j1e_coef = x[2*n_nuc:]

    print(' env expo: {}'.format(x[:n_nuc]))
    print(' j1e expo: {}'.format(x[n_nuc:]))
    print(' j1e coef: {}'.format(x[n_nuc:]))

    h = str(x)
    if h in globals.memo_energy:
        return globals.memo_energy[h]

    globals.i_fev = globals.i_fev + 1

    # UPDATE PARAMETERS
    set_env_expo(env_expo, atom_map, ezfio)
    set_j1e_expo(j1e_expo, j1e_size, atom_map, ezfio)
    set_j1e_coef(j1e_expo, j1e_size, atom_map, ezfio)

    # OPTIMIZE ORBITALS
    e_tcscf, c_tcscf = run_tcscf(ezfio, EZFIO_file)
    clear_tcscf_orbitals(EZFIO_file)
    if c_tcscf:
        mohf = np.array(ezfio.get_mo_basis_mo_coef()).T
        mor  = np.array(ezfio.get_bi_ortho_mos_mo_r_coef()).T
        ezfio.set_mo_basis_mo_coef(mor.T)
    else:
        return 100.0 + 10.0 * random.random()

    # GET VMC energy & variance 
    set_vmc_params(block_time_f, total_time_f, EZFIO_file)

    loc_err = 10.
    ii      = 1
    ii_max  = 5 
    energy  = None
    err     = None
    while(Eloc_err_th < loc_err):

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

    return energy + var_weight * var_en

# ---

