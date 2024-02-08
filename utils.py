
import subprocess
import globals
from globals import block_time_f, total_time_f, Eloc_err_th, var_weight



# ---

def run_qmc(EZFIO_file):
    return subprocess.check_output(['qmcchem', 'run', EZFIO_file])

# ---

def stop_qmc(EZFIO_file):
    subprocess.check_output(['qmcchem', 'stop', EZFIO_file])

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

def f(x):

    print('\n eval {} of f on:'.format(globals.i_fev))
    print(' nuc param Jast = {}'.format(x[:-1]))
    print(' b   param Jast = {}'.format(x[-1]))

    h = str(x)
    if h in globals.memo_energy:
        return globals.memo_energy[h]

    globals.i_fev = globals.i_fev + 1

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

        elif( globals.memo_energy['fmin'] < (energy-2.*err) ):
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

    return energy + var_weight * var_en

# ---





