from autograd import elementwise_grad as egrad
from autograd import hessian
import autograd.numpy as np
#import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import date
### add pyneb
sys.path.insert(0, '../../py_neb/')
sys.path.insert(0, '../../../flynn_code/py_neb_demos')
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

LAP_path = np.loadtxt('../../../Paths/CAMEL_BACK/PyNeb_6_camel_back_symm_LAP_path.txt',\
                      delimiter=',',skiprows=1)
exact_LAP_evals = np.loadtxt('../../../py_neb/examples/6_camel_back/symm_camel_exact_evals.txt',\
                      delimiter='\t')
H_f = hessian(camel_back_xy)
test_pnt = np.array([0.0,0.0])

diff = np.zeros((len(LAP_path),2))
autograd_evals = np.zeros((len(LAP_path),2))
for i,pnt in enumerate(LAP_path):
    hess = H_f(pnt)
    autograd_evals[i] = np.linalg.eigvals(hess)
    diff[i] = autograd_evals[i] - exact_LAP_evals[i]


plt.plot(LAP_path[:,0],autograd_evals[:,0],label='autograd 0')
plt.plot(LAP_path[:,0],autograd_evals[:,1],label='autograd 1')
#plt.plot(LAP_path[:,0],exact_LAP_evals[:,0],label='exact 0')
#plt.plot(LAP_path[:,0],exact_LAP_evals[:,1],label='exact 1')
plt.title('evals x')
plt.legend()
plt.xlim([-1,1])
plt.show()
#plt.plot(LAP_path[:,0],exact_LAP_evals[:,0])
#plt.xlim([-.5,.5])
#plt.title('exact evals x')
#plt.show()

