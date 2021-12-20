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

def get_crit_pnts(V_func,path):
    '''
    NOTE: This function depends on a package called autograd for hessian calculation
    
    This function finds the critical the MEP path must pass through by first finding the 
    critical points of the energy along the curve and then classifies them
    using the eigenvalues of the Hessian. Returns minima, maxima, and saddle points indices
    along the path.
    
    Parameters
    ----------
    V_func : object
        Energy Function that must have shape (nImgs,nDims).
    path : ndarray
        coordinates of the path on the surface with shape (nImgs,nDims).
    interpolate : Boolean, optional
        Interpolate path using nImgs (default 500) . The default is True.
    nImgs : int, optional
        Number of images to interpolate path to. The default is 500.

    Returns
    -------
    3 arrays containing the indices of minima, maxima, and saddle points.

    '''
    ### path should be shape (nImgs,nDims)
    nDim = path.shape[1]
    H = hessian(V_func)
    EnergyOnPath = V_func(path)
    minima_pnts = utilities.SurfaceUtils.find_all_local_minimum(EnergyOnPath)[0]
    maxima_pnts = utilities.SurfaceUtils.find_all_local_maxima(EnergyOnPath)[0]
    crit_pnts = np.concatenate((minima_pnts,maxima_pnts))
    maxima = []
    minima = []
    saddle = []
    for indx in crit_pnts:
        coord = interp_path[indx]
        hess = H(coord)
        evals = np.linalg.eigvals(hess)
        ## see which components are less than 0.
        neg_bool = evals < 0
        ## count how many falses there are (ie how many postives there are)
        eval_num = np.count_nonzero(neg_bool)
        if eval_num == 0:
            # if all evals are positive, then H is positive def and the function is
            # concave up at this point. This means we are at a local minima
            minima.append(indx)
        elif eval_num == nDim:
            # if all evals are positive, then H is negative def and the function is
            # concave down at this point. This means we are at a local maxima
            maxima.append(indx)
        else:
            # if evals are positive and negative, 
            # this means we are at a local saddle
            saddle.append(indx)
    return(maxima,minima,saddle)

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

MEP_path = np.loadtxt('../../Paths/CAMEL_BACK/PyNeb_6_camel_back_symm_MEP_path.txt',\
                      delimiter=',',skiprows=1)
exact_LAP_evals = np.loadtxt('./symm_camel_exact_evals.txt',\
                      delimiter='\t')

nImgs = 500
path_call = utilities.InterpolatedPath(MEP_path)
t_array = np.linspace(0,1,nImgs)
interp_path = np.array(path_call(t_array)).T
maxima,minima,saddle = get_crit_pnts(camel_back,interp_path)
print(minima)
print(maxima)
print(saddle)
plt.plot(interp_path[:,0],interp_path[:,1])
plt.plot(interp_path[:,0][saddle],interp_path[:,1][saddle],'o')
plt.plot(interp_path[:,0][minima],interp_path[:,1][minima],'o')
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