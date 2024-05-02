
import numpy as np

from globals import ezfio

from utils.atoms import atom_map


# ---

def get_params_pen():
    d = ezfio.jastrow_jast_pen
    return np.array([d[a[0]] for a in atom_map])

def set_params_pen(x):
    y = list(ezfio.jastrow_jast_pen)
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_jast_pen(y)

# ---

def get_params_b():
    b = ezfio.get_jastrow_jast_b_up_up()
    return b

def set_params_b(b):
    ezfio.set_jastrow_jast_b_up_up(b)
    ezfio.set_jastrow_jast_b_up_dn(b)

# ---

def get_mu():
    return ezfio.get_hamiltonian_mu_erf()

def set_mu(mu):
    ezfio.set_hamiltonian_mu_erf(mu)

# ---

def get_env_coef():
    d = ezfio.get_jastrow_env_coef()
    return np.array([d[a[0]] for a in atom_map])
def set_env_coef(x):
    y = list(ezfio.get_jastrow_env_coef())
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_env_coef(y)

def get_env_expo():
    d = ezfio.get_jastrow_env_expo()
    return np.array([d[a[0]] for a in atom_map])
def set_env_expo(x):
    y = list(ezfio.get_jastrow_env_expo())
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_env_expo(y)

# ---

def get_j1e_size():
    return ezfio.get_jastrow_j1e_size()
def set_j1e_size(j1e_size):
    ezfio.set_jastrow_j1e_size(j1e_size)

def get_j1e_coef(j1e_size):
    f = np.array(ezfio.get_jastrow_j1e_coef()).T
    r = []
    for a in atom_map:
        for p in range(j1e_size):
            r.append(f[p,a[0]])
    return r
def set_j1e_coef(x, j1e_size):
    f = np.array(ezfio.get_jastrow_j1e_coef()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(j1e_size):
                f[p,j] = x[i*j1e_size+p]
    ezfio.set_jastrow_j1e_coef(f.T)

def get_j1e_expo(j1e_size):
    f = np.array(ezfio.get_jastrow_j1e_expo()).T
    r = []
    for a in atom_map:
        for p in range(j1e_size):
            r.append(f[p,a[0]])
    return r
def set_j1e_expo(x, j1e_size):
    f = np.array(ezfio.get_jastrow_j1e_expo()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(j1e_size):
                f[p,j] = x[i*j1e_size+p]
    ezfio.set_jastrow_j1e_expo(f.T)

# ---

