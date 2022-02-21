import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import os
### add pyneb
sys.path.insert(0, '../py_neb/py_neb')
sys.path.insert(0, '../flynn_code/py_neb_demos')
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
    result = 3.17181 + 2*np.exp(-5*((x-1)**2 + (y-.5)**2)) -3*np.exp(-(x**2 +y**2)) -.5*(3*x +y)
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
RN_array = [[1.1,1.4],[.8,1.4] ,[.3,1.5],[1.06521739,0.58361204],[1.5,0],[1.75,-.25],[2.0,-.4]]
LAP_array = []
actions = []