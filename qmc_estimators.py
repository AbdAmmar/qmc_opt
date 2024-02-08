
import subprocess



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

def set_vmc_params(block_time, total_time, EZFIO_file):
    subprocess.check_output([ 'qmcchem', 'edit', '-c'
                            #, '-j', 'Simple'
                            , '-t', str(total_time)
                            , '-l', str(block_time)
                            , EZFIO_file] )

# ---

d
