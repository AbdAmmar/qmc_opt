
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
                            , '-t', str(globals.vmc_total_time)
                            , '-l', str(globals.vmc_block_time)
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

def get_var_Htc():
    buffer = subprocess.check_output(['qmcchem', 'result', globals.EZFIO_file], encoding='UTF-8')
    if buffer.strip() != "":
        lines = buffer.split('\n')
        for line in lines:
            if "Var_htc" in line:
                parts = line.split()
                break
        return float(parts[2]), float(parts[4])
    else:
        return None, None

# ---

