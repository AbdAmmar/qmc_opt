
import time
import numpy as np
import sys
import random
import globals
from scipy.optimize import rosen
import random

# ---

def f_rosen(x, *args):

    h = str(x)
    if h in globals.memo_res:
        return globals.memo_res[h]

    print('\n eval {} of f on:'.format(globals.i_fev))
    print('\n x = {}'.format(x))

    energy = rosen(x)
    var_en = 0.0
    err    = 0.0

    print(" energy = {} ".format(energy))

    sys.stdout.flush()
    #time.sleep(3) 

    globals.i_fev = globals.i_fev + 1
    globals.memo_res[h]      = energy + err
    globals.memo_res['fmin'] = min(energy, globals.memo_res['fmin'])

    return energy + globals.var_weight * var_en

# ---

def init_rosen(n_par):

    b = 1e+2

    x, x_min, x_max = [], [], []
    for _ in range(n_par):
        xx = 2*b*random.random() - b
        x.append(xx)
        x_min.append(-1.0*b)
        x_max.append(+1.0*b)

    return x, x_min, x_max

# ---

