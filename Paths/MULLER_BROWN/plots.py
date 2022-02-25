import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

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
### Plot title
plt_title = 'MB_plot.pdf'

### Paths to plot
# './PyNeb_6_camel_back_symm_LAP_path.txt'
LAP_path = np.loadtxt('./PyNeb_muller_brown__LAP_path.txt',skiprows=1,delimiter=',')
MEP_path = np.loadtxt('./PyNeb_muller_brown__MEP_path.txt',skiprows=1,delimiter=',')
DP_path = np.loadtxt('./mullerbrown_Path.txt',delimiter=',',skiprows=1)
EL_path = np.loadtxt('./PathELEMuler-Brown.csv',delimiter=',')
DJK_path = np.loadtxt("./dijkstra.txt",delimiter=",")
print(LAP_path[-1],MEP_path[-1],DP_path[0],EL_path[-1])
paths = {'NEB-LAP': LAP_path,'NEB-MEP': MEP_path,'DP': DP_path,'EL':EL_path,\
         "Dijkstra":DJK_path}
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

maxima,minima,saddle = utilities.get_crit_pnts(V_func, MEP_path,method='central')

### Plot the results

fig, mainAx = plt.subplots(1,1)
axins = zoomed_inset_axes(mainAx,5)

for ax in [mainAx,axins]:
    im = ax.contourf(grids[0],grids[1],EE.clip(0,195),cmap='Spectral_r',extend='both',levels=45)
    cs = ax.contour(grids[0],grids[1],EE.clip(0,195),colors=['black'],levels=7,linewidths=.5)  
    #cs = ax.contour(grids[0],grids[1],EE.clip(0,195),colors=['black'],levels=np.arange(0,195,50),linewidths=.5) 
    #ax.contour(grids[0],grids[1],EE.clip(40,195),colors=['black'],levels=np.arange(40,195,50),linewidths=.5) 
    colorsDict = {'DP':"black","EL":"cyan","NEB-LAP":"magenta","NEB-MEP":"red","Dijkstra":"lime"}
    for key, path in paths.items():
        ax.plot(*path.T,label=key,linestyle="-",color=colorsDict[key],linewidth=1.0)    
    ax.clabel(cs,inline=1,fontsize=6,colors="black")
    ax.plot(MEP_path[:,0][saddle],MEP_path[:,1][saddle],'*',color='black',markersize=8)
    ax.plot(MEP_path[:,0][minima],MEP_path[:,1][minima],'X',color='yellow',markersize=6)
    
    
axins.set_xlim(-0.65, -0.45)
axins.set_ylim(1.3, 1.5)
axins.set_xticklabels([])
axins.set_yticklabels([])

axins.axvline(MEP_path[:,0][minima][0],ls="--",color="yellow")

mark_inset(mainAx,axins, edgecolor="black",loc1=2,loc2=3)

#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#mainAx.tick_params(labelsize=14)
mainAx.set_xlabel(r'$x$')
mainAx.set_ylabel(r'$y$')

cbar = fig.colorbar(im,ax=mainAx)
#cbar.ax.tick_params(labelsize=14) 
#plt.setp(mainAx.get_yticklabels()[0], visible=False)    
#plt.setp(mainAx.get_xticklabels()[0], visible=False)  
plt.savefig(plt_title,bbox_inches="tight")
plt.show()
