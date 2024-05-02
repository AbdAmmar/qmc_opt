
import subprocess
import os
import globals


# ---

def clear_tcscf_orbitals(ezfio):
    mor = os.path.join(globals.EZFIO_file, "bi_ortho_mos", "mo_r_coef.gz")
    mol = os.path.join(globals.EZFIO_file, "bi_ortho_mos", "mo_l_coef.gz")
    etc = os.path.join(globals.EZFIO_file, "tc_scf", "bitc_energy")
    if os.path.exists(mor):
        subprocess.check_call(['rm', mor])
    if os.path.exists(mol):
        subprocess.check_call(['rm', mol])
    if os.path.exists(etc):
        subprocess.check_call(['rm', etc])
    ezfio.set_tc_scf_converged_tcscf(False)

def run_tcscf(ezfio):
    with open("tc_scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'tc_scf', globals.EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    etc = os.path.join(globals.EZFIO_file, "tc_scf", "bitc_energy")
    ctc = os.path.join(globals.EZFIO_file, "tc_scf", "converged_tcscf")
    e_tcscf = None
    c_tcscf = False
    if os.path.exists(etc):
        e_tcscf = ezfio.get_tc_scf_bitc_energy()
    if os.path.exists(ctc):
        c_tcscf = ezfio.get_tc_scf_converged_tcscf()
    return e_tcscf, c_tcscf

# ---

def run_scf(ezfio):
    with open("scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'scf', globals.EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    e_scf = ezfio.get_hartree_fock_energy()
    return e_scf

# ---

