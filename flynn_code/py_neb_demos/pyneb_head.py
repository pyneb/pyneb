import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
from matplotlib.pyplot import cm

from datetime import date
import sys
import pandas as pd
### add pyneb
sys.path.insert(0, '../../py_neb')
import py_neb
import utils
import re


nucleus = "232U"
data_path = f"../../PES/{nucleus}.h5"
PES = utils.PES(data_path)
mass_PES = utils.PES(data_path)
#PES_test.get_data_shapes()
uniq_coords = PES.get_unique(return_type='array')
grids,EE,E_gs,gs_coord = PES.get_grids(return_coord_grid=True,shift_GS=True,ignore_val=-1760)
mass_grids = PES.get_mass_grids()
mass_keys = mass_grids.keys()
mass_tensor_indicies = ['20','30']

# TODO: add section to handle PES shifts.

#Define potential function
# note this interpolator only interpolates points or arrays of points, no grids.
V_func = py_neb.GridInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential',minVal=0)
# Create dictionary of functions for each comp of the tensor
mass_grids = {key: py_neb.GridInterpWithBoundary(uniq_coords,mass_grids[key],boundaryHandler='exponential',minVal=0) \
              for key in mass_keys}
# function returns matrix of functions
M_func = py_neb.mass_funcs_to_array_func(mass_grids,mass_tensor_indicies)

NImgs = 22
NIterations = 10
k = 10.0
kappa = 20.0
E_const = E_gs
nDims = len(uniq_coords)
fix_R0 = True
fix_RN = True
springR0 = False
springRN = False
endPointFix = (fix_R0,fix_RN)
springForceFix = (springR0,springRN)


target_func = py_neb.action # function you want to minimize
target_func_grad = py_neb.forward_action_grad ## specialized function that takes the gradient of target function

neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}

LAP_params = {'potential':V_func,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func,\
                 'target_func_grad':target_func_grad,'nebParams':neb_params}


lap = py_neb.LeastActionPath(**LAP_params)
minObj = py_neb.VerletMinimization(lap)
'''
minObj.minimize_however(tstep=0.1,maxIters=100)
'''


