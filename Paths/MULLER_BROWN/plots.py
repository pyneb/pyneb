import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys

### add pyneb
import sys
sys.path.insert(0, '../../py_neb/py_neb')
sys.path.insert(0, '../../flynn_code/py_neb_demos')
import utilities
import utils
plt.style.use('science')
def make_MB_potential():
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
        #print(x)
        #print(y)
        for i in range(len(A)):
            result += A[i]*np.exp(a[i]*(x-x_bar[i])**2.0 + b[i]*(x-x_bar[i])*(y-y_bar[i]) + c[i]*(y-y_bar[i])**2.0)
        return result
    return MB_potential
### Plot title
plt_title = 'MB_plot_v1.pdf'

### Paths to plot
# './PyNeb_6_camel_back_symm_LAP_path.txt'
LAP_path = np.loadtxt('./PyNeb_muller_brown__LAP_path.txt',skiprows=1,delimiter=',')
MEP_path = np.loadtxt('./PyNeb_muller_brown__MEP_path.txt',skiprows=1,delimiter=',')
DP_path = np.loadtxt('./mullerbrown_DPM_Path.txt',skiprows=1,delimiter=',')
paths = {'LAP': LAP_path,'MEP': MEP_path,'DP': DP_path}
path_names = paths.keys()
action_dict = {}
#Define potential function
V_func = make_MB_potential()

### Shift Surface by global minimum
x = np.linspace(-1.5, 1,300)
y = np.linspace(-.25,1.9,300)
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
V_func_shift = utilities.shift_func(V_func,shift=E_gs)#shift by the ground state
EE = V_func_shift(np.array([xx,yy]))

interp_paths = {}

## Interpolate the paths 
for name in path_names:
    path = paths[name]
    path_call = utilities.InterpolatedPath(path)
    action_dict[name] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift])[1][0],3)

with open('MB_action_values.txt', 'w+') as f:
    for name in path_names:
        f.write(str(name)+': '+str(action_dict[name])+'\n')
for name in action_dict.keys():
    print(name+': ', action_dict[name])
    
### Plot the results
fig, ax = plt.subplots(1,1,figsize = (8, 6))
im = ax.contourf(grids[0],grids[1],EE.clip(0,195),cmap='Spectral_r',extend='both',levels=45)
cs = ax.contour(grids[0],grids[1],EE.clip(0,195),colors=['black'],levels=10)  

ax.plot(DP_path[:, 0], DP_path[:, 1],label='DP',linestyle='-',color='black',linewidth=2.0)
ax.plot(EL_path[:, 0], EL_path[:, 1],label='EL',linestyle='-',color='cyan',linewidth=2.0)
ax.plot(LAP_path[:, 0], LAP_path[:, 1],label='NEB-LAP ',linestyle='-',color='magenta',linewidth=2.0)
ax.plot(MEP_path[:, 0], MEP_path[:, 1],label='NEB-MEP ',linestyle='-',color='red',linewidth=2.0)

ax.plot(LAP_path[0][0],LAP_path[0][1],marker='s',color='yellow',markersize=5)
ax.plot(LAP_path[-1][0],LAP_path[-1][1],marker='s',color='yellow',markersize=5)

ax.clabel(cs,inline=1,fontsize=8,colors="black")

cbar = fig.colorbar(im)
cbar.ax.tick_params(labelsize=12) 
plt.legend(frameon=True,fancybox=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set(xlabel="x",ylabel="y")
#plt.xlabel(r'$x$',size=24)
#plt.ylabel(r'$y$',size=24)
plt.savefig(plt_title,bbox_inches="tight")
plt.show()
