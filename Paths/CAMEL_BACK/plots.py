import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys

### add pyneb
sys.path.insert(0, '../../py_neb/py_neb')
sys.path.insert(0, '../../flynn_code/py_neb_demos')
import utilities
import utils
plt.style.use('science')
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
    result = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2) 
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
    result = (4 - 2.1*(x**2) + 1/3 * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2) + .5*y
    return(result)

### Plot title
plt_title = 'camel_plot_symm_v1.pdf'

### Paths to plot
LAP_path = np.loadtxt('./PyNeb_6_camel_back_symm_LAP_path.txt',skiprows=1,delimiter=',')
MEP_path = np.loadtxt('./PyNeb_6_camel_back_symm_MEP_path.txt',skiprows=1,delimiter=',')
DP_path = np.loadtxt('./CAMEL_symmetric_DPM_Path.txt',skiprows=1,delimiter=',')
paths = {'LAP': LAP_path,'MEP': MEP_path,'DP': DP_path}
path_names = paths.keys()
action_dict = {}
#Define potential function
V_func = camel_back_asymm

### Find global minimum to shift surface
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
####
gs_coord = minima_coords[0]
E_gs = V_func(gs_coord)
print('E_gs: ',E_gs)
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))




## Interpolate the paths 
for name in path_names:
    path = paths[name]
    path_call = utilities.InterpolatedPath(path)
    action_dict[name] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)

for name in action_dict.keys():
    print(name+': ', action_dict[name])
x = np.linspace(-2, 2,300)
y = np.linspace(-1.25, 1.25,300)
uniq_coords = [x,y]

xx,yy = np.meshgrid(x,y)
grids = [xx,yy]
coords = np.array([xx,yy])

EE = V_func(np.array([xx,yy]))





fig, ax = plt.subplots(1,1,figsize = (8, 6))
im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,4))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=MaxNLocator(nbins = 15).tick_values(0,4))  
ax.plot(LAP_path[:, 0], LAP_path[:, 1],label='LAP '+str(action_dict['LAP']),linestyle='-',marker='.',color='purple',linewidth=1.0,markersize=6)
ax.plot(MEP_path[:, 0], MEP_path[:, 1],label='MEP '+str(action_dict['MEP']),linestyle='-',marker='.',color='red',linewidth=1.0,markersize=6)
ax.plot(DP_path[:, 0], DP_path[:, 1],label='DPM '+str(action_dict['DP']),linestyle='-',marker='.',color='black',linewidth=1.0,markersize=6)
cbar = fig.colorbar(im)
plt.legend(frameon=True,fancybox=True)
plt.savefig(plt_title)
plt.show()
