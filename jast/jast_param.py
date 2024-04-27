
import numpy as np

# ---

def get_params_pen(atom_map, ezfio):
    d = ezfio.jastrow_jast_pen
    return np.array([d[a[0]] for a in atom_map])

def set_params_pen(x, atom_map, ezfio):
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

def get_mu(ezfio):
    return ezfio.get_hamiltonian_mu_erf()

def set_mu(mu, ezfio):
    ezfio.set_hamiltonian_mu_erf(mu)

# ---

def get_env_coef(atom_map, ezfio):
    d = ezfio.get_jastrow_env_coef()
    return np.array([d[a[0]] for a in atom_map])
def set_env_coef(x, atom_map, ezfio):
    y = list(ezfio.get_jastrow_env_coef())
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_env_coef(y)

def get_env_expo(atom_map, ezfio):
    d = ezfio.get_jastrow_env_expo()
    return np.array([d[a[0]] for a in atom_map])
def set_env_expo(x, atom_map, ezfio):
    y = list(ezfio.get_jastrow_env_expo())
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_env_expo(y)

# ---

def get_j1e_size(ezfio):
    return ezfio.get_jastrow_j1e_size()
def set_j1e_size(j1e_size, ezfio):
    ezfio.set_jastrow_j1e_size(j1e_size)

def get_j1e_coef(j1e_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_coef()).T
    r = []
    for a in atom_map:
        for p in range(j1e_size):
            r.append(f[p,a[0]])
    return r
def set_j1e_coef(x, j1e_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_coef()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(j1e_size):
                f[p,j] = x[i*j1e_size+p]
    ezfio.set_jastrow_j1e_coef(f.T)

def get_j1e_expo(j1e_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_expo()).T
    r = []
    for a in atom_map:
        for p in range(j1e_size):
            r.append(f[p,a[0]])
    return r
def set_j1e_expo(x, j1e_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_expo()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(j1e_size):
                f[p,j] = x[i*j1e_size+p]
    ezfio.set_jastrow_j1e_expo(f.T)

# ---

def get_jbh_size(ezfio):
    return ezfio.get_jastrow_jbh_size()
def set_jbh_size(jbh_size, ezfio):
    ezfio.set_jastrow_jbh_size(jbh_size)

def get_jbh_c(jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_c()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_c(x, jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_c()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_c(f.T)

def get_jbh_m(jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_m()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_m(x, jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_m()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_m(f.T)

def get_jbh_n(jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_n()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_n(x, jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_n()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_n(f.T)

def get_jbh_o(jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_o()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_o(x, jbh_size, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_jbh_o()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_o(f.T)

def get_jbh_ee(atom_map, ezfio):
    d = ezfio.get_jastrow_jbh_ee()
    return np.array([d[a[0]] for a in atom_map])
def set_jbh_ee(x, atom_map, ezfio):
    y = list(ezfio.jastrow_jbe_ee)
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_jbh_ee(y)

def get_jbh_en(atom_map, ezfio):
    d = ezfio.get_jastrow_jbh_en()
    return np.array([d[a[0]] for a in atom_map])
def set_jbh_en(x, atom_map, ezfio):
    y = list(ezfio.jastrow_jbe_en)
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_jbh_en(y)

# ---



