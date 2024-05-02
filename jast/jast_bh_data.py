
# ---

def init_jbh9_water(atom_map, ezfio):

    jbh_size = 9

    jbh_m = [[0, 0, 0, 0, 2, 3, 4, 2, 2], [0, 0, 0, 0, 2, 3, 4, 2, 2], [0, 0, 0, 0, 2, 3, 4, 2, 2]]
    jbh_n = [[0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0]]
    jbh_o = [[1, 2, 3, 4, 0, 0, 0, 0, 2], [1, 2, 3, 4, 0, 0, 0, 0, 2], [1, 2, 3, 4, 0, 0, 0, 0, 2]]
    jbh_c = [[+0.5, -0.57070, +0.49861, -0.78663, +0.01990, +0.13386, -0.60446, -1.67160, +1.36590],
             [+0.0, +0.0    , +0.0    , +0.0    , +0.12063, -0.18527, +0.12324, -0.11187, -0.06558],
             [+0.0, +0.0    , +0.0    , +0.0    , +0.12063, -0.18527, +0.12324, -0.11187, -0.06558]]

    jbh_ee = [1.0, 1.0, 1.0]
    jbh_en = [1.0, 1.0, 1.0]

    ezfio.set_jastrow_jbh_size(jbh_size)

    ezfio.set_jastrow_jbh_o(jbh_o)
    ezfio.set_jastrow_jbh_m(jbh_m)
    ezfio.set_jastrow_jbh_n(jbh_n)
    ezfio.set_jastrow_jbh_c(jbh_c)

    ezfio.set_jastrow_jbh_ee(jbh_ee)
    ezfio.set_jastrow_jbh_en(jbh_en)

# ---

