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
def make_MB_potential():
    # parameter set taken from 1701.01241 (scaled down)
    A = [-200,-100,-170,15]
    a = [-1,-1,-6.5,0.7]
    b= [0,0,11,0.6]
    c = [-10,-10,-6.5,0.7]
    x_bar = [1,0,-0.5,-1]
    y_bar = [0,0.5,1.5,1]
    def MB_potential(coords):
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
        result = np.zeros(x.shape)
        for i in range(len(A)):
            result += A[i]*np.exp(a[i]*(x-x_bar[i])**2 + b[i]*(x-x_bar[i])*(y-y_bar[i]) + c[i]*(y-y_bar[i])**2)
        return result
    return MB_potential
surface_name = "muller_brown_" # for output files
#Define potential function
save_data = False
V_func = make_MB_potential()
M_func = None # for LAP 
auxFunc = None # for MEP

x = np.linspace(-1.5, 1,300)
y = np.linspace(-.25,2,300)
uniq_coords = [x,y]

xx,yy = np.meshgrid(x,y)
grids = [xx,yy]
coords = np.array([xx,yy])
EE_0 = V_func(np.array([xx,yy]))

#Calculate minima of PES
minima_id = utilities.SurfaceUtils.find_all_local_minimum(EE_0)
minima_coords = np.array((xx[minima_id],yy[minima_id])).T
V_min = V_func(minima_coords)
sorted_ascending_idx = np.argsort(V_min) #places global min first

# redefine minima_coords so that they are in ascending order
minima_coords = np.array(minima_coords[sorted_ascending_idx])
V_min = V_func([minima_coords[:,0],minima_coords[:,1]])

#########
gs_coord = minima_coords[0]
E_gs = V_func(gs_coord)
V_func_shift = utilities.shift_func(V_func,shift=E_gs)# shift by the ground state

EE = V_func_shift(np.array([xx,yy]))

#
NImgs = 32 # number of images
k = 3.0 # spring constant for entire band
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
NIterations = 300

### define initial path
# NEB starting point
R0 = [-0.55, 1.44]
# NEB end point
RN = [0.62, 0.03]

## make initial path guess
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

### Begining the optimization procedure. 
t0 = time.time()
tStepArr, alphaArr, stepsSinceReset, endsWithoutError = minObj_LAP.fire(dt,NIterations,useLocal=False,earlyStop=False)
allPaths_LAP = minObj_LAP.allPts
final_path_LAP = allPaths_LAP[-1]

t1 = time.time()
total_time_LAP = t1 - t0
print('total_time LAP: ',total_time_LAP)


action_array_LAP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_LAP):
    ## computer action over interpolated path using 500 points
    path_call = utilities.InterpolatedPath(path)
    action_array_LAP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)
min_action_LAP = np.around(action_array_LAP[-1],3)




#### Compute MEP
# MEP function you want to minimize
target_func_MEP = utilities.TargetFunctions.mep_default
# MEP specialized function that takes the gradient of target function
target_func_grad_MEP = utilities.potential_central_grad 
MEP_params = {'potential':V_func_shift,'nPts':NImgs,'nDims':nDims,'auxFunc':auxFunc,'endpointSpringForce': springForceFix ,\
                 'endpointHarmonicForce':endPointFix,'target_func':target_func_MEP,\
                 'target_func_grad':target_func_grad_MEP,'nebParams':neb_params}
t0 = time.time()
mep = solvers.MinimumEnergyPath(**MEP_params)
minObj_MEP = solvers.VerletMinimization(mep,initialPoints=init_path)
tStepArr, alphaArr, stepsSinceReset, endsWithoutError = minObj_MEP.fire(dt,NIterations,useLocal=False,earlyStop=False)
allPaths_MEP = minObj_MEP.allPts

t1 = time.time()
total_time_MEP = t1 - t0
final_path_MEP = allPaths_MEP[-1]
print('Total time MEP: ',total_time_MEP)

action_array_MEP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_MEP):
    ## computer action over interpolated path over 500 points
    path_call = utilities.InterpolatedPath(path)
    action_array_MEP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,1000,tfArgs=[V_func_shift])[1][0],3)
min_action_MEP =  np.around(action_array_MEP[-1],3)

### Find stationary points on domain using MEP
maxima,minima,saddle = utilities.get_crit_pnts(V_func, final_path_MEP,method='central')


# Plot the results.
### Make contour plot of surface
fig, ax = plt.subplots(1,1)
im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,215))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 12).tick_values(0,215))  
ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'green',ms=10,label='Initial Path')
ax.plot(final_path_LAP[:, 0], final_path_LAP[:, 1], '.-',ms=10,label='LAP',color='purple')
ax.plot(final_path_MEP[:, 0], final_path_MEP[:, 1], '.-',ms=10,label='MEP',color='red')    
ax.plot(final_path_MEP[:,0][saddle],final_path_MEP[:,1][saddle],'*',color='black',markersize=12)
ax.plot(final_path_MEP[:,0][minima],final_path_MEP[:,1][minima],'X',color='yellow',markersize=12)


ax.set_ylabel('$y$',size=20)
ax.set_xlabel('$x$',size=20)
ax.set_title(f'N = {NImgs}, k= {k}, kappa= {kappa}')
ax.legend(frameon=True,fancybox=True)
cbar = fig.colorbar(im)
plt.show()

### Plot the actions
plt.plot(range(NIterations+2),action_array_LAP,label='Final LAP '+str(min_action_LAP))
plt.plot(range(NIterations+2),action_array_MEP,label='Final MEP '+str(min_action_MEP))
plt.xlabel('Iterations')
plt.ylabel('Action')
plt.legend(frameon=True,fancybox=True)
plt.show()