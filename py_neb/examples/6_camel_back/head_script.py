import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys
from datetime import date
### add pyneb
sys.path.insert(0, '../../py_neb/')
sys.path.insert(0, '../../../flynn_code/py_neb_demos')
import solvers
import utilities
import utils
today = date.today()
def camel_back(coords):
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
def camel_back_broken_sym(coords):
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

surface_name = "6_camel_back_symm" # for output files
save_data = False
#Define potential function
V_func = camel_back
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
print('E_gs: ',E_gs)
print(minima_coords)
print(V_func(minima_coords))
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))



####

NImgs = 52# number of images
k = 1.0 # spring constant for entire band
kappa = 1.0 # harmonic force strength (only used if force_RN or force_R0 is True)
E_const = E_gs # energy contour to constrain the end points to
nDims = len(uniq_coords)
force_R0 = False
force_RN = True
springR0 = False
springRN = True
endPointFix = (force_R0,force_RN)
springForceFix = (springR0,springRN)

### Optimization parameters 
## Velocity Verlet parameter set
dt = .05
NIterations = 2000


### define initial path
#beginning point 
R0 = [0,0]
#R0 = [-1.700 ,0.790]# NEB starting point for symmetric case
#R0 = [1.700 ,-0.800]# NEB starting point for asymmetric case
#end point
#RN = [0.08695652, -0.71488294]
RN = [-0.08695652, -0.71488294]
#RN =  [1.700 , -0.790]# NEB end point for symmetric case
#RN =  [-1.700 , 0.760]# NEB end point for asymmetric case
print('R0: ',R0)
print('RN: ',RN)
init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
init_path = init_path_constructor.linear_path()
#init_path = init_path_constructor.linear_path()
### Define parameter dictionaries (mostly for book keeping)
neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}
method_dict = {'k':k,'kappa':kappa,'NImages': NImgs,'Iterations':NIterations,'dt':dt,'optimization':'Local FIRE','HarmonicForceEnds': endPointFix, \
                   'SpringForceEnds': springForceFix,'logLevel':0}

    
    
      
#### Compute LAP
# LAP function you want to minimize
target_func_LAP = utilities.TargetFunctions.action
# LAP specialized function that takes the gradient of target function
target_func_grad_LAP = utilities.GradientApproximations().forward_action_grad

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
tStepArr, alphaArr, stepsSinceReset = minObj_LAP.fire(dt,NIterations,useLocal=True)
allPaths_LAP = minObj_LAP.allPts
final_path_LAP = allPaths_LAP[-1]


t1 = time.time()
total_time_LAP = t1 - t0
print('total_time LAP: ',total_time_LAP)
action_array_LAP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_LAP):
    action_array_LAP[i] = utilities.TargetFunctions.action(path, V_func_shift,M_func)[0]
min_action_LAP = np.around(action_array_LAP[-1],4)

## Save metadata
title = 'PyNeb_'+surface_name+'_LAP'

metadata = {'title':title,'Created_by': 'Eric','Created_on':today.strftime("%b-%d-%Y"),'method':'NEB-LAP','method_description':method_dict, \
                'masses':None,'E_gs': str(E_gs),'action':action_array_LAP[-1],'run_time':total_time_LAP ,\
                    'initial_start_point': R0,'initial_end_point': RN}
utils.make_metadata(metadata)

# write final path to txt.
if save_data == True:
    np.savetxt(title+'_path.txt',final_path_LAP,comments='',delimiter=',',header="x,y")








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
tStepArr, alphaArr, stepsSinceReset = minObj_MEP.fire(dt,NIterations,useLocal=True)
allPaths_MEP = minObj_MEP.allPts

t1 = time.time()
total_time_MEP = t1 - t0
final_path_MEP = allPaths_MEP[-1]
print('total_time MEP: ',total_time_MEP)
### Compute the action of each path in allPts_MEP
action_array_MEP = np.zeros(NIterations+2)
for i,path in enumerate(allPaths_MEP):
    action_array_MEP[i] = utilities.TargetFunctions.action(path, V_func_shift ,None)[0]   # endPointFix = (force_R0,force_RN) springForceFix
min_action_MEP =  np.around(action_array_MEP[-1],4)


## Save metadata
title = 'PyNeb_'+surface_name+'_MEP'

metadata = {'title':title,'Created_by': 'Eric','Created_on':today.strftime("%b-%d-%Y"),'method':'NEB-MEP','method_description':method_dict, \
                'masses':None,'E_gs': str(E_gs),'action':action_array_MEP[-1],'run_time':total_time_MEP ,\
                    'initial_start_point': R0,'initial_end_point': RN}
utils.make_metadata(metadata)

if save_data == True:
    np.savetxt(title+'_path.txt',final_path_MEP,comments='',delimiter=',',header="Q20,Q30")
### Plot the results.
fig, ax = plt.subplots(1,1,figsize = (12, 10))

im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,4))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 15).tick_values(0,4))  
ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'green',ms=10,label='Initial Path')
ax.plot(final_path_LAP[:, 0], final_path_LAP[:, 1], '.-',ms=10,label='LAP',color='purple')
ax.plot(final_path_MEP[:, 0], final_path_MEP[:, 1], '.-',ms=10,label='MEP',color='red')    


ax.set_ylabel('$y$',size=20)
ax.set_xlabel('$x$',size=20)
ax.set_title('M = '+str(NIterations)+' N = '+str(NImgs)+' k='+str(k)+' kappa='+str(kappa))


ax.legend(frameon=True,fancybox=True)
cbar = fig.colorbar(im)
if save_data ==True:
    plt.savefig(surface_name+'_M='+str(NIterations)+'_N='+str(NImgs)+'_k='+str(k)+'_kappa='+str(kappa)+'.pdf')
plt.show()  
plt.clf()

plt.plot(range(NIterations+2),action_array_LAP,label='LAP '+str(min_action_LAP))
plt.plot(range(NIterations+2),action_array_MEP,label='MEP '+str(min_action_MEP))
plt.xlabel('Iterations')
plt.ylabel('Action')
plt.legend(frameon=True,fancybox=True)
if save_data == True:
    plt.savefig(surface_name+'_M='+str(NIterations)+'_N='+str(NImgs)+'_k='+str(k)+'_kappa='+str(kappa)+'_action.pdf')
plt.show()
print('completed Nimg= '+str(NImgs))