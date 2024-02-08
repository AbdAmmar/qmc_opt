
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

def set_j1e_size(n, ezfio):
    ezfio.set_jastrow_j1e_size(n)

def get_j1e_coef(n, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_coef()).T
    r = []
    for a in atom_map
        for i in range(n):
            r.append(f[i,a[0]])
    return r

def set_j1e_coef(x, n, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_coef()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(n):
                f[p,j] = x[i+p]
    ezfio.set_jastrow_env_coef(f)

def get_j1e_expo(n, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_expo()).T
    r = []
    for a in atom_map
        for i in range(n):
            r.append(f[i,a[0]])
    return r

def set_j1e_expo(x, n, atom_map, ezfio):
    f = np.array(ezfio.get_jastrow_j1e_expo()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(n):
                f[p,j] = x[i+p]
    ezfio.set_jastrow_env_expo(f)

# ---


