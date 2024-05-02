
import numpy as np

from globals import ezfio

from utils.atoms import atom_map


# ---

def get_jbh_size():
    return ezfio.get_jastrow_jbh_size()
def set_jbh_size(jbh_size):
    ezfio.set_jastrow_jbh_size(jbh_size)

def get_jbh_c(jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_c()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_c(x, jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_c()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_c(f.T)

def get_jbh_m(jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_m()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_m(x, jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_m()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_m(f.T)

def get_jbh_n(jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_n()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_n(x, jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_n()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_n(f.T)

def get_jbh_o(jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_o()).T
    r = []
    for a in atom_map:
        for p in range(jbh_size):
            r.append(f[p,a[0]])
    return r
def set_jbh_o(x, jbh_size):
    f = np.array(ezfio.get_jastrow_jbh_o()).T
    for i,a in enumerate(atom_map):
        for j in a:
            for p in range(jbh_size):
                f[p,j] = x[i*jbh_size+p]
    ezfio.set_jastrow_jbh_o(f.T)

def get_jbh_ee():
    d = ezfio.get_jastrow_jbh_ee()
    return np.array([d[a[0]] for a in atom_map])
def set_jbh_ee(x):
    y = list(ezfio.jastrow_jbe_ee)
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_jbh_ee(y)

def get_jbh_en():
    d = ezfio.get_jastrow_jbh_en()
    return np.array([d[a[0]] for a in atom_map])
def set_jbh_en(x):
    y = list(ezfio.jastrow_jbe_en)
    for i,a in enumerate(atom_map):
        for j in a:
            y[j] = x[i]
    ezfio.set_jastrow_jbh_en(y)

# ---



