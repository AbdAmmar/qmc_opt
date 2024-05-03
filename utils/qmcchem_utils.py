
import subprocess

from globals import EZFIO_file
from globals import vmc_total_time, vmc_block_time


# ---

def run_qmc():
    return subprocess.check_output(['qmcchem', 'run', EZFIO_file])

# ---

def stop_qmc():
    subprocess.check_output(['qmcchem', 'stop', EZFIO_file])

# ---

def set_vmc_params():
    subprocess.check_output([ 'qmcchem', 'edit', '-c'
                            , '-t', str(vmc_total_time)
                            , '-l', str(vmc_block_time)
                            , EZFIO_file] )

# ---

def get_energy():
    buffer = subprocess.check_output(
            ['qmcchem', 'result', '-e', 'e_loc', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, energy, error = [float(x) for x in buffer.split()]
        return energy, error
    else:
        return None, None

# ---

def get_variance():
    buffer = subprocess.check_output(['qmcchem', 'result', '-e', 'e_loc_qmcvar', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        buffer = buffer.splitlines()[-1]
        _, variance, error = [float(x) for x in buffer.split()]
        return variance, error
    else:
        return None, None

# ---

def get_QMC_scalar(scalar_name):
    buffer = subprocess.check_output(['qmcchem', 'result', EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        lines = buffer.split('\n')
        for line in lines:
            if scalar_name in line:
                parts = line.split()
                break
        return float(parts[2]), float(parts[4])
    else:
        return None, None

# ---

