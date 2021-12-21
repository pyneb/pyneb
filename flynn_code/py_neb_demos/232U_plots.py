import numpy as np
import matplotlib.pyplot as plt
import utils
import sys
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, '../../py_neb/py_neb')
import utilities
nucleus = "232U"
data_path = f"../../PES/{nucleus}.h5"
### defines PES object from utils.py

PES = utils.PES(data_path)
mass_PES = utils.PES(data_path)
uniq_coords = PES.get_unique(return_type='array')
grids,EE = PES.get_grids(return_coord_grid=True,shift_GS=False)
mass_grids = PES.get_mass_grids()
mass_keys = mass_grids.keys()
### IMPORTANT: LIST THE INDICIES OF THE MASS TENSOR TO USE.
mass_tensor_indicies = ['20','30']

minima_ind = utilities.SurfaceUtils.find_all_local_minimum(EE)
gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.25,0.25],returnOnlySmallest=True)
gs_coord = np.array((grids[0][gs_ind],grids[1][gs_ind])).T
E_gs_raw = EE[gs_ind]
EE = EE - E_gs_raw


V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential',symmExtend=[True,True])

E_gs = V_func(gs_coord)


LAP_nomass = np.loadtxt('../../Paths/232U/Eric_232U_LAP_Mass_False_path.txt',delimiter=',',skiprows=1)
MEP_nomass = np.loadtxt('../../Paths/232U/Eric_232U_MEP_Mass_True_path.txt',delimiter=',',skiprows=1)
LAP_mass = np.loadtxt('../../Paths/232U/Eric_232U_LAP_Mass_True_path.txt',delimiter=',',skiprows=1)

#test = np.array([False,False])
#print(np.count_nonzero(test))
maxima,minima,saddle = utilities.get_crit_pnts(V_func,MEP_nomass,method='central')
print(minima)
print(maxima)
print(saddle)

fig, ax = plt.subplots(1,1,figsize = (12, 10))
im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,15))
ax.contour(grids[0],grids[1],EE,colors=['black'],levels=[E_gs])              
#ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'orange',ms=10,label='Initial Path')
ax.plot(LAP_nomass[:,0], LAP_nomass[:,1], '.-',ms=10,label='LAP_NoMass',color='purple')
ax.plot(LAP_mass[:,0], LAP_mass[:,1], '.-',ms=10,label='LAP_Mass',color='orange')
ax.plot(MEP_nomass[:,0], MEP_nomass[:,1], '.-',ms=10,label='MEP_NoMass',color='red')  
ax.plot(MEP_nomass[:,0][saddle], MEP_nomass[:,1][saddle],'*',ms=18,color='cyan')  
ax.set_ylabel('$Q_{30}$',size=20)
ax.set_xlabel('$Q_{20}$',size=20)
ax.set_title('240Pu')
ax.legend()
cbar = fig.colorbar(im)
#plt.savefig(nucleus+'.pdf')
plt.show()  
plt.clf()