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
today = date.today()
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


#Define potential function
# note this interpolator only interpolates points or arrays of points, no grids.

V_func = py_neb.NDInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential',minVal=0)
# Create dictionary of functions for each comp of the tensor
mass_grids = {key: py_neb.NDInterpWithBoundary(uniq_coords,mass_grids[key],boundaryHandler='exponential',minVal=0) \
              for key in mass_keys}
# function returns matrix of functions
#M_func = py_neb.mass_funcs_to_array_func(mass_grids,mass_tensor_indicies)
M_func = None # for LAP 
auxFunc = None # for MEP

NImgs = 42
k = 2.0
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
dt = .02
NIterations = 100


### define initial path
R0 = gs_coord # NEB starting point
RN = (295,30) # NEB end point
init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
init_path = init_path_constructor.linear_path()

### Define parameter dictionaries (mostly for book keeping)
neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}
method_dict = {'k':k,'kappa':kappa,'NImages': NImgs,'Iterations':NIterations,'dt':dt,'optimization':'QM Verlet','HarmonicForceEnds': endPointFix, \
                   'SpringForceEnds': springForceFix}
#### Compute LAP
'''
# LAP function you want to minimize
target_func_LAP = py_neb.action 
# LAP specialized function that takes the gradient of target function
target_func_grad_LAP = py_neb.forward_action_grad 

LAP_params = {'potential':V_func,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_LAP,\
                 'target_func_grad':target_func_grad_LAP,'nebParams':neb_params}

### define the least action object 
### This essentially defines the forces given the target and gradient functions 
lap = py_neb.LeastActionPath(**LAP_params)

### Define the optimizer object to use. Note the initial band is passed
### here and the operations defined in LeastActionPath are applied to
### the band.
minObj_LAP = py_neb.VerletMinimization(lap,initialPoints=init_path)

### Begining the optimization procedure. Results are all of the velocities
### band positions, and forces for each iteration of the optimization.
allPaths_LAP, allVelocities_LAP, allForces_LAP = minObj_LAP.velocity_verlet(dt,NIterations)
final_path_LAP = allPaths_LAP[-1]
print('finished LAP')
action_array_LAP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_LAP):
    action_array_LAP[i] = py_neb.action(path, V_func,M_func)[0]
min_action_LAP = np.around(min(action_array_LAP),2)
title = 'Eric_'+nucleus+'_LAP'
'''

#### Compute MEP
# MEP function you want to minimize
target_func_MEP = py_neb.potential_target_func
# MEP specialized function that takes the gradient of target function
target_func_grad_MEP = py_neb.potential_central_grad 
MEP_params = {'potential':V_func,'nPts':NImgs,'nDims':nDims,'auxFunc':auxFunc,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_MEP,\
                 'target_func_grad':target_func_grad_MEP,'nebParams':neb_params}
MEP_time1 = time.time()
mep = py_neb.MinimumEnergyPath(**MEP_params)
minObj_MEP = py_neb.VerletMinimization(mep,initialPoints=init_path)
allPaths_MEP, allVelocities_MEP, allForces_MEP = minObj_MEP.velocity_verlet(dt,NIterations)
MEP_time2 = time.time()
total_time_MEP = MEP_time2 - MEP_time1
final_path_MEP = allPaths_MEP[-1]
title = 'Eric_'+nucleus+'_MEP'
print('total_time: ',total_time_MEP)



### Compute the action of each path in allPts

action_array_MEP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_MEP):
    action_array_MEP[i] = py_neb.action(path, V_func,None)[0]   # endPointFix = (force_R0,force_RN) springForceFix
min_action_MEP =  np.around(min(action_array_MEP),2)


'''
metadata = {'title':title,'Created_by': 'Eric','Created_on':today.strftime("%b-%d-%Y"),'method':'NEB-MEP','method_description':method_dict, \
                'masses':None,'E_gs': str(E_gs),'action':action_array_MEP[-1],'run_time':total_time_MEP ,\
                    'initial_start_point': R0,'initial_end_point': RN}
utils.make_metadata(metadata)
## should include plot title, method, date created, creator, action value, wall time
    ## model description {k: 10, kappa: 20, nPts: 22, nIterations: 750, optimization: velocity_verlet, endpointForce: on}
np.savetxt(title+'_path.txt',final_path_MEP,comments='',delimiter=',',header="Q20,Q30")
'''

### Plot the results.
fig, ax = plt.subplots(1,1,figsize = (12, 10))

im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(-2,15))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=[E_gs])              
ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'green',ms=10,label='Initial Path')
#ax.plot(final_path_LAP[:, 0], final_path_LAP[:, 1], '.-',ms=10,label='LAP')
ax.plot(final_path_MEP[:, 0], final_path_MEP[:, 1], '.-',ms=10,label='MEP')    
ax.set_ylabel('$Q_{30}$',size=20)
ax.set_xlabel('$Q_{20}$',size=20)
ax.set_title('M = '+str(NIterations)+' N = '+str(NImgs)+' k='+str(k)+' kappa='+str(kappa))
ax.legend()
cbar = fig.colorbar(im)
#plt.savefig('M = '+str(NIterations)+' N = '+str(NImgs)+' k='+str(k)+' kappa='+str(kappa)+'.pdf')
plt.show()  
plt.clf()

#plt.plot(range(NIterations+2),action_array_LAP,label='LAP '+str(min_action_LAP))
plt.plot(range(NIterations+2),action_array_MEP,label='MEP '+str(min_action_MEP))
plt.xlabel('Iterations')
plt.ylabel('Action')
plt.legend()
plt.show()


