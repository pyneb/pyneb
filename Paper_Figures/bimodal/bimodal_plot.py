import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import sys
import os
### add pyneb
sys.path.insert(0, '../../py_neb/py_neb')
sys.path.insert(0, '../../flynn_code/py_neb_demos')
import solvers
import utilities
import utils
plt.style.use('science')

def test_func(coords):
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
    #3.17181
    result = 3.17 + 2*np.exp(-5*((x-1)**2 + (y-.5)**2)) -3*np.exp(-(x**2 +y**2)) -.5*(3*x +y)
    return(result)


movie_name = "multi-modal" # for output files
dir_name = movie_name+'-frames'
os.makedirs(dir_name, exist_ok=True) 
#Define potential function
V_func = test_func
M_func = None # for LAP 
auxFunc = None # for MEP

x = np.linspace(-.5, 2.5,300)
y = np.linspace(-.5, 2.5,300)
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
print(minima_coords)
#########
gs_coord = minima_coords[0]
E_gs = V_func(np.array([0.21 ,.0607]))
print('E_gs: ',E_gs)
print(minima_coords)
print(V_func(minima_coords))
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))



####

NImgs =32# number of images
k = 4.0 # spring constant for entire band
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
dt = .1
NIterations = 400


### define initial path
#beginning point 
RN_array = [[1.06521739,0.58361204],[1.5,0]]
LAP_array = []
actions = []
for RN in RN_array:
    
    R0 = [0.21 ,.0607]# NEB starting point for asymmetric case
    #RN =  [1.1,1.4]# NEB end point for symmetric case
    
    print('R0: ',R0)
    print('RN: ',RN)
    init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
    init_path = init_path_constructor.linear_path()
    #init_path = init_path_constructor.linear_path()
    ### Define parameter dictionaries (mostly for book keeping)
    neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}
    method_dict = {'k':k,'kappa':kappa,'NImages': NImgs,'Iterations':NIterations,'dt':dt,'optimization':'FIRE2','HarmonicForceEnds': endPointFix, \
                       'SpringForceEnds': springForceFix,'logLevel':0}
    
    FireParams = {"dtMax":1.,"dtMin":10**(-2),"nAccel":5,"fInc":1.1,"fAlpha":.5,\
         "fDecel":0.8,"aStart":0.1,"maxmove":np.array([.1,.1,1])}
        
          
    #### Compute LAP
    # LAP function you want to minimize
    target_func_LAP = utilities.TargetFunctions.action
    # LAP specialized function that takes the gradient of target functionpath
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
    tStepArr, alphaArr, stepsSinceReset = minObj_LAP.fire2(dt,NIterations,fireParams=FireParams,useLocal=False)
    allPaths_LAP= minObj_LAP.allPts
    final_path_LAP = allPaths_LAP[-1]
    LAP_array.append(allPaths_LAP) 
    
    action_array_LAP = np.zeros(NIterations+2)
    for i,path in enumerate(allPaths_LAP):
        #action_array_LAP[i] = utilities.TargetFunctions.action(path, V_func_shift,M_func)[0]
        path_call = utilities.InterpolatedPath(path)
        action_array_LAP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)
    min_action_LAP = np.around(action_array_LAP[-1],4)
    actions.append(action_array_LAP)

### Plot the results
fig, ax = plt.subplots(1,1,figsize = (12, 10))

axins = zoomed_inset_axes(ax,3,loc=2)


im = axins.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,2.5))
cs = axins.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 15).tick_values(0.01,2.5))  
axins.plot(LAP_array[0][-1][:, 0], LAP_array[0][-1][:, 1], '-',label="{:.3f}".format(actions[0][-1]),ms=18,linewidth=5,color='black')
axins.plot(LAP_array[1][-1][:, 0], LAP_array[1][-1][:, 1], '-',label="{:.3f}".format(actions[1][-1]),ms=18,linewidth=5,color='cyan')
axins.plot(LAP_array[0][60][:, 0], LAP_array[0][60][:, 1], '.-',label="{:.3f}".format(actions[0][-1]),ms=18,linewidth=5,color='gray')
axins.contour(grids[0],grids[1],EE,colors=['white'],levels=[E_gs],linestyles='solid',linewidths=8)  

im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,2.5))
cs = ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 15).tick_values(0.01,2.5))  
ax.plot(LAP_array[0][-1][:, 0], LAP_array[0][-1][:, 1], '-',label="{:.3f}".format(actions[0][-1]),ms=18,linewidth=5,color='black')
ax.plot(LAP_array[1][-1][:, 0], LAP_array[1][-1][:, 1], '-',label="{:.3f}".format(actions[1][-1]),ms=18,linewidth=5,color='cyan')
ax.plot(LAP_array[0][60][:, 0], LAP_array[0][60][:, 1], '-',label="{:.3f}".format(actions[0][-1]),ms=18,linewidth=5,color='gray')
ax.contour(grids[0],grids[1],EE,colors=['white'],levels=[E_gs],linestyles='solid',linewidths=8)  

axins.set_xlim(0.65, 1.1)
axins.set_ylim(.4, .8)
axins.set_xticklabels([])
axins.set_yticklabels([])
mark_inset(ax,axins, edgecolor="black",loc1=1,loc2=3)
plt.setp(ax.get_yticklabels()[0], visible=False)    
plt.setp(ax.get_xticklabels()[0], visible=False)  
ax.set_ylabel('$y$',size=32)
ax.set_xlabel('$x$',size=32)
#ax.legend(frameon=True,fancybox=True,fontsize=20)
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24) 

cbar = fig.colorbar(im,ax=ax,format='%.1f')
cbar.ax.tick_params(labelsize=24) 


plt.savefig('bimodal_bare.pdf')
plt.show()
plt.clf()
plt.close()