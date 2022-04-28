import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys
### add pyneb
sys.path.insert(0, '../../src/pyneb')
import solvers
import utilities
def linear_path(R0,RN,NImgs):
    ## returns the initial positions of every point on the chain.
    path = np.zeros((NImgs,len(R0)))
    for i in range(len(R0)):
        xi = np.linspace(R0[i],RN[i],NImgs)
        path[:,i] = xi
    return(path)

def camel_back_symm(coords):
    if isinstance(coords, np.ndarray)==False:
        coords = np.array(coords)
    ## check if it's scalar
    if len(coords.shape) == 1:
        coords = coords.reshape(1,-1)
        x = coords[:,0]
        y = coords[:,1]
    else:pass
    if len(coords.shape) < 3:
        x = coords[:,0]
        y = coords[:,1]
    else:pass
    if len(coords.shape)>= 3:
        x = coords[0]
        y = coords[1]
    else:pass
    result = (4 - 2.1*(x**2) + (1/3) * (x**4))*(x**2) + x*y + 4*((y**2) - 1)*(y**2) 
    return(result)
def camel_back_asymm(coords):
    if isinstance(coords, np.ndarray)==False:
        coords = np.array(coords)
    ## check if it's scalar
    if len(coords.shape) == 1:
        coords = coords.reshape(1,-1)
        x = coords[:,0]
        y = coords[:,1]
    else:pass
    if len(coords.shape) < 3:
        x = coords[:,0]
        y = coords[:,1]
    else:pass
    if len(coords.shape)>= 3:
        x = coords[0]
        y = coords[1]
    else:pass
    result = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2) + .5*y
    return(result)

#Define potential function. To switch between symmetric and asymmetric
# just change the V_func name to the function you want
V_func = camel_back_asymm
M_func = None # for LAP 
auxFunc = None # for MEP

x = np.linspace(-2, 2,300)
y = np.linspace(-1.25, 1.25,300)
uniq_coords = [x,y]

xx,yy = np.meshgrid(x,y)
grids = [xx,yy]
coords = np.array([xx,yy])
EE_0 = V_func(np.array([xx,yy]))

#Calculate minima of PES
minima_id =  utilities.SurfaceUtils.find_all_local_minimum(EE_0)
minima_coords = np.array((xx[minima_id],yy[minima_id])).T
V_min = V_func(minima_coords)
sorted_ascending_idx = np.argsort(V_min) #places global min first

# redefine minima_coords so that they are in ascending order
minima_coords = minima_coords[sorted_ascending_idx]
V_min = V_func([minima_coords[:,0],minima_coords[:,1]])

#########
gs_coord = minima_coords[0]
E_gs = V_func(gs_coord)
print(f'Ground State Location: {gs_coord}')
print(f'Ground State Energy: {E_gs}')
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))



####

NImgs = 52# number of images
k = 2.0 # spring constant for entire band
kappa = 1.0 # harmonic force strength (only used if force_RN or force_R0 is True)
E_const = E_gs # energy contour to constrain the end points to
nDims = len(uniq_coords)
force_R0 = False
force_RN = False
springR0 = False
springRN = False
endPointFix = (force_R0,force_RN)
springForceFix = (springR0,springRN)

### Optimization parameters 
## Velocity Verlet parameter set
dt = .01
NIterations = 800


### define initial path
#beginning point 
R0 = [-1.700 ,0.790]# NEB starting point for symmetric case
#R0 = [1.700 ,-0.800]# NEB starting point for asymmetric case
#end point
RN =  [1.700 , -0.790]# NEB end point for symmetric case
#RN =  [-1.700 , 0.760]# NEB end point for asymmetric case

init_path = linear_path(R0,RN,NImgs)
### Define parameter dictionaries (mostly for book keeping)
neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}
    
      
#### Compute LAP
# LAP function you want to minimize
target_func_LAP = utilities.TargetFunctions.action
# LAP specialized function that takes the gradient of target function
target_func_grad_LAP = utilities.GradientApproximations().discrete_action_grad_const

LAP_params = {'potential':V_func_shift,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_LAP,\
                 'target_func_grad':target_func_grad_LAP,'nebParams':neb_params}

### define the least action object 
### This essentially defines the forces given the target and gradient functions 
lap = solvers.LeastActionPath(**LAP_params)

### Define the optimizer object to use. Note the initial band is passed
### here and the operations defined in LeastActionPath are applied to
### the band.
minObj_LAP = solvers.VerletMinimization(lap,initialPoints=init_path)

### Begining the optimization procedure. Results are all of the velocities
### band positions, and forces for each iteration of the optimization.
t0 = time.time()
tStepArr, alphaArr, stepsSinceReset = minObj_LAP.fire2(dt,NIterations,useLocal=False,earlyStop=False)
allPaths_LAP = minObj_LAP.allPts
final_path_LAP = allPaths_LAP[-1]


t1 = time.time()
total_time_LAP = t1 - t0
print('Total Time LAP: ',total_time_LAP)
action_array_LAP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_LAP):
    ## computer action over interpolated path using 500 points
    path_call = utilities.InterpolatedPath(path)
    action_array_LAP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)
min_action_LAP = np.around(action_array_LAP[-1],4)



#### Compute MEP
# MEP function you want to minimize
target_func_MEP = utilities.TargetFunctions.mep_default
# MEP specialized function that takes the gradient of target function
target_func_grad_MEP = utilities.potential_central_grad 
MEP_params = {'potential':V_func,'nPts':NImgs,'nDims':nDims,'auxFunc':auxFunc,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_MEP,\
                 'target_func_grad':target_func_grad_MEP,'nebParams':neb_params}
t0 = time.time()
mep = solvers.MinimumEnergyPath(**MEP_params)
minObj_MEP = solvers.VerletMinimization(mep,initialPoints=init_path)
tStepArr, alphaArr, stepsSinceReset = minObj_MEP.fire2(dt,NIterations,useLocal=False,earlyStop=False)
allPaths_MEP = minObj_MEP.allPts

t1 = time.time()
total_time_MEP = t1 - t0
final_path_MEP = allPaths_MEP[-1]
print('Total Time MEP: ',total_time_MEP)
### Compute the action of each path in allPts_MEP
action_array_MEP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_MEP):
    ## computer action over interpolated path using 500 points
    path_call = utilities.InterpolatedPath(path)
    action_array_MEP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)
min_action_MEP =  np.around(action_array_MEP[-1],4)

### Find stationary points on domain using the MEP
maxima,minima,saddle = utilities.get_crit_pnts(V_func, final_path_MEP,method='central')


# Plot the results.
### Make contour plot
fig, ax = plt.subplots(1,1)
im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,4))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 8).tick_values(0,4))  
ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'green',ms=10,label='Initial Path')
ax.plot(final_path_LAP[:, 0], final_path_LAP[:, 1], '.-',ms=10,label='LAP',color='purple')
ax.plot(final_path_MEP[:, 0], final_path_MEP[:, 1], '.-',ms=10,label='MEP',color='red')    
ax.plot(final_path_MEP[:,0][saddle],final_path_MEP[:,1][saddle],'*',color='black',markersize=12)
ax.plot(final_path_MEP[:,0][minima],final_path_MEP[:,1][minima],'X',color='yellow',markersize=12)
ax.set_ylabel('$y$',size=20)
ax.set_xlabel('$x$',size=20)
ax.set_title(f'N = {NImgs}, k= {k}, kappa={kappa}')

ax.legend(frameon=True,fancybox=True)
cbar = fig.colorbar(im)
plt.show()  
plt.clf()

### Make Action plot
plt.plot(range(NIterations+2),action_array_LAP,label='Final LAP '+str(min_action_LAP))
plt.plot(range(NIterations+2),action_array_MEP,label='Final MEP '+str(min_action_MEP))
plt.xlabel('Iterations')
plt.ylabel('Action')
plt.legend(frameon=True,fancybox=True)
plt.show()