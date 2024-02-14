
import subprocess
import globals


# ---

def run_qmc(EZFIO_file):
    return subprocess.check_output(['qmcchem', 'run', EZFIO_file])

# ---

def stop_qmc(EZFIO_file):
    subprocess.check_output(['qmcchem', 'stop', EZFIO_file])

# ---

def set_vmc_params(block_time, total_time, EZFIO_file):
    subprocess.check_output([ 'qmcchem', 'edit', '-c'
                            , '-j', globals.j2e_type
                            , '-t', str(total_time)
                            , '-l', str(block_time)
                            , EZFIO_file] )

# ---

def get_energy(EZFIO_file):
    buffer = subprocess.check_output(
            ['qmcchem', 'result', '-e', 'e_loc', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, energy, error = [float(x) for x in buffer.split()]
        return energy, error
    else:
        return None, None

# ---

def get_variance(EZFIO_file):
    buffer = subprocess.check_output(['qmcchem', 'result', '-e', 'e_loc_qmcvar', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, variance, error = [float(x) for x in buffer.split()]
        return variance, error
    else:
        return None, None

# ---

