
import subprocess



# ---

def run_qmc(EZFIO_file):
    return subprocess.check_output(['qmcchem', 'run', EZFIO_file])

# ---

def stop_qmc(EZFIO_file):
    subprocess.check_output(['qmcchem', 'stop', EZFIO_file])

# ---

