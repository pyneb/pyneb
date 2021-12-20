from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import hessian
import autograd.numpy as np
#import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
### add pyneb
sys.path.insert(0, '../../py_neb/py_neb')
sys.path.insert(0, '../py_neb_demos')
import solvers
import utilities
import utils
today = date.today()
def test_func(coords):
    x = coords[0]
    y = coords[1]
    result = x**2 + 2*y**2 + x*y
    return(result)
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
def camel_back_xy(coords):
    x = coords[0]
    y = coords[1]
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
def path_wrapper():
    return
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
V_func_shift = utilities.shift_func(V_func,shift=E_gs) #shift by the ground state
EE = V_func_shift(np.array([xx,yy]))


## Import path

LAP_path = np.loadtxt('../../Paths/CAMEL_BACK/PyNeb_6_camel_back_symm_LAP_path.txt',\
                      delimiter=',',skiprows=1)
exact_LAP_evals = np.loadtxt('./symm_camel_exact_evals.txt',\
                      delimiter='\t')

path_call = utilities.InterpolatedPath(LAP_path)
nImages = 500
t_array = np.linspace(0,1,nImages)
interp_path = np.array(path_call(t_array)).T
grad_V = grad(camel_back_xy)
H_f = hessian(camel_back_xy)

path_V = camel_back(interp_path)
minima_pnts = utilities.SurfaceUtils.find_all_local_minimum(path_V)[0]
maxima_pnts = utilities.SurfaceUtils.find_all_local_maxima(path_V)[0]
crit_pnts = np.concatenate((minima_pnts,maxima_pnts))
maxima = []
minima = []
saddle = []
for pnt in crit_pnts:
    coord = interp_path[pnt]
    nDim = coord.shape[0]
    hess = H_f(coord)
    evals = np.linalg.eigvals(hess)
    ## see which components are less than 0.
    neg_bool = evals < 0
    ## count how many falses there are (ie how many postives there are)
    eval_num = np.count_nonzero(neg_bool)
    if eval_num == 0:
        # if all evals are positive, then H is positive def
        # This means we are at a local maximum
        maxima.append([coord,pnt])
    elif eval_num == nDim:
        # if all evals are negative, then H is negative def
        # This means we are at a local minimum
        minima.append([coord,pnt])
    else:
        # if evals are positive and negative, 
        # this means we are at a local saddle
        saddle.append([coord,pnt])
print(saddle)
plt.plot(t_array,camel_back(interp_path))
plt.plot(t_array[crit_pnts],camel_back(interp_path[crit_pnts]),'o')
plt.show()


'''
plt.plot(autograd_evals[:,1],label='autograd 1')
plt.title('evals 1')
plt.legend()
#plt.xlim([-1,1])
plt.show()

plt.plot(autograd_evals[:,0],'o',label='autograd 0')
#plt.plot(autograd_evals[:,1],'o',label='autograd 1')
plt.plot(exact_LAP_evals[:,0],label='exact 0')
#plt.plot(exact_LAP_evals[:,1],label='exact 1')

plt.title('evals 1')
plt.legend()
#plt.xlim([-1,1])
plt.show()
'''