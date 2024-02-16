
import subprocess
import globals


# ---

def run_qmc():
    return subprocess.check_output(['qmcchem', 'run', globals.EZFIO_file])

# ---

def stop_qmc():
    subprocess.check_output(['qmcchem', 'stop', globals.EZFIO_file])

# ---

def set_vmc_params():
    subprocess.check_output([ 'qmcchem', 'edit', '-c'
                            , '-j', globals.j2e_type
                            , '-t', str(globals.total_time)
                            , '-l', str(globals.block_time)
                            , globals.EZFIO_file] )

# ---

def get_energy():
    buffer = subprocess.check_output(
            ['qmcchem', 'result', '-e', 'e_loc', globals.EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, energy, error = [float(x) for x in buffer.split()]
        return energy, error
    else:
        return None, None

# ---

def get_variance():
    buffer = subprocess.check_output(['qmcchem', 'result', '-e', 'e_loc_qmcvar', globals.EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, variance, error = [float(x) for x in buffer.split()]
        return variance, error
    else:
        return None, None

# ---

