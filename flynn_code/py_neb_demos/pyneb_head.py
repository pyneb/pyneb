import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
from matplotlib.pyplot import cm

from datetime import date
import sys
import pandas as pd
### add pyneb
#sys.path.insert(0, '../../py_neb')
import py_neb_temp
import utils

    
nucleus = "232U"
data_path = f"../../PES/{nucleus}.h5"
PES = utils.PES(data_path)
mass_keys=['B2020','B2030','B2030','B3030']
mass_PES = utils.PES(data_path)
#PES_test.get_data_shapes()
uniq_coords = PES.get_unique(return_type='array')
grids,EE,gs_coord = PES.get_grids(return_coord_grid=True,shift_GS=True,ignore_val=-1760)
mass_grids = PES.get_mass_grids()
grids[0].shape
#Define potential function
# note this interpolator only interpolates points or arrays of points

V_func = py_neb_temp.GridInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential',minVal=0)
# Define mass tensor
M_func = py_neb_temp.make_mass_tensor(uniq_coords, mass_grids, mass_keys)


'''
lap = LeastActionPath(*args,**kwargs)
minObj = VerletMinimization(lap)

minObj.minimize_however(tstep=0.1,maxIters=100)
'''


