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

### Define nucleus data path (assumes our github structure)
nucleus = "232U"
data_path = f"../../PES/{nucleus}.h5"
### defines PES object from utils.py
PES = utils.PES(data_path)
mass_PES = utils.PES(data_path)
uniq_coords = PES.get_unique(return_type='array')
grids,EE,E_gs,gs_coord = PES.get_grids(return_coord_grid=True,shift_GS=True,ignore_val=-1760)
mass_grids = PES.get_mass_grids()
mass_keys = mass_grids.keys()
### IMPORTANT: LIST THE INDICIES OF THE MASS TENSOR TO USE.
mass_tensor_indicies = ['20','30']

# TODO: add section to handle PES shifts.

#Define potential function
# note this interpolator only interpolates points or arrays of points, no grids.
V_func = py_neb.GridInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential',minVal=0)
# Create dictionary of functions for each comp of the tensor
mass_grids = {key: py_neb.GridInterpWithBoundary(uniq_coords,mass_grids[key],boundaryHandler='exponential',minVal=0) \
              for key in mass_keys}
# function returns matrix of functions
#M_func = py_neb.mass_funcs_to_array_func(mass_grids,mass_tensor_indicies)
M_func = None


NImgs = 32
k = 15.0
kappa = 20.0
E_const = E_gs
nDims = len(uniq_coords)
force_R0 = False
force_RN = True
springR0 = False
springRN = True
endPointFix = (force_R0,force_RN)
springForceFix = (springR0,springRN)

### Optimization parameters 
## Velocity Verlet parameter set
dt = .01
NIterations = 500

target_func = py_neb.action # function you want to minimize
target_func_grad = py_neb.forward_action_grad ## specialized function that takes the gradient of target function

### define initial path
R0 = gs_coord # NEB starting point
RN = (290,32) # NEB end point
init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
init_path = init_path_constructor.linear_path()

### Define parameter dictionaries (mostly for book keeping)
neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}

LAP_params = {'potential':V_func,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func,\
                 'target_func_grad':target_func_grad,'nebParams':neb_params}

### define the least action object 
### This essentially defines the forces given the target and gradient functions 
lap = py_neb.LeastActionPath(**LAP_params)

### Define the optimizer object to use. Note the initial band is passed
### here and the operations defined in LeastActionPath are applied to
### the band.
minObj = py_neb.VerletMinimization(lap,initialPoints=init_path)

### Beging the optimization procedure. Results are all of the velocities
### band positions, and forces for each iteration of the optimization.
allPaths, allVelocities, allForces = minObj.velocity_verlet(dt,NIterations)
final_path = allPaths[-1]
### Compute the action of each path in allPts
action_array = np.zeros(NIterations+1)
for i,path in enumerate(allPaths):
    action_array[i] = py_neb.action(path, V_func,M_func)[0]

### Plot the results.
fig, ax = plt.subplots(1,1,figsize = (12, 10))

im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(-2,15))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=[E_gs])              
ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'green',ms=10,label='Initial Path')
ax.plot(final_path[:, 0], final_path[:, 1], '.-',ms=10)
    
ax.set_ylabel('$Q_{30}$',size=20)
ax.set_xlabel('$Q_{20}$',size=20)
ax.set_title('M = '+str(NIterations)+' N = '+str(NImgs)+' k='+str(k)+' kappa='+str(kappa))
ax.legend()
cbar = fig.colorbar(im)
#plt.savefig('M = '+str(NIterations)+' N = '+str(NImgs)+' k='+str(k)+' kappa='+str(kappa)+'.pdf')
plt.show()  
plt.clf()
min_action = np.around(min(action_array),2)
plt.plot(range(NIterations+1),action_array,label=str(min_action))
plt.xlabel('Iterations')
plt.ylabel('Action')
plt.show()


