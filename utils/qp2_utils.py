
import subprocess
import os

import globals
from globals import ezfio, EZFIO_file


# ---

def clear_tcscf_orbitals():
    mor = os.path.join(EZFIO_file, "bi_ortho_mos", "mo_r_coef.gz")
    mol = os.path.join(EZFIO_file, "bi_ortho_mos", "mo_l_coef.gz")
    etc = os.path.join(EZFIO_file, "tc_scf", "bitc_energy")
    if os.path.exists(mor):
        subprocess.check_call(['rm', mor])
    if os.path.exists(mol):
        subprocess.check_call(['rm', mol])
    if os.path.exists(etc):
        subprocess.check_call(['rm', etc])
    ezfio.set_tc_scf_converged_tcscf(False)

# ---

def run_tcscf():
    with open("tc_scf.out", "w") as f:
        subprocess.check_call(['qp_run', 'tc_scf', EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    etc = os.path.join(EZFIO_file, "tc_scf", "bitc_energy")
    ctc = os.path.join(EZFIO_file, "tc_scf", "converged_tcscf")
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

def run_gs_tc_energy(j2e, env, j1e, io_integ, type_integ):
    ezfio.set_jastrow_j2e_type(j2e)
    ezfio.set_jastrow_env_type(env)
    ezfio.set_jastrow_j1e_type(j1e)
    ezfio.set_tc_keywords_io_tc_integ(io_integ)
    ezfio.set_tc_keywords_tc_integ_type(type_integ)
    with open("E_TC.out", "w") as f:
        subprocess.check_call(['qp_run', 'print_tc_energy', EZFIO_file], stdout=f, stderr=subprocess.STDOUT)
    e_tc = ezfio.get_tc_bi_ortho_tc_gs_energy()
    return e_tc

# ---

