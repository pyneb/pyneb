import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import os
import glob 
import sys
sys.path.insert(0, '../../../py_neb/py_neb')
sys.path.insert(0, '../../../flynn_code/py_neb_demos/')

import utilities
import utils
# Import path data 
home = os.path.expanduser('~')
path_files = sorted(glob.glob(home+'/ActionMinimization/Paths/240Pu/*.txt'))
for path_file in path_files:
    directory,file = os.path.split(path_file)
    method_name = file.split('-')
    
    
## Import PES 
nucleus = "240Pu"
data_path = f"../../../PES/{nucleus}_new.h5"
use_mass = False
save_data = True
save_plt = False
### defines PES object from utils.py
PES = utils.PES(data_path)
mass_PES = utils.PES(data_path)
uniq_coords = PES.get_unique(return_type='array')

grids,EE = PES.get_grids(return_coord_grid=True,shift_GS=False)
mass_grids = PES.get_mass_grids()
mass_keys = mass_grids.keys()
### IMPORTANT: LIST THE INDICIES OF THE MASS TENSOR TO USE.
mass_tensor_indicies = ['20','30','pair']

minima_ind = utilities.SurfaceUtils.find_all_local_minimum(EE)
gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.2,0.2,0.2],returnOnlySmallest=False)
gs_coord = np.array((grids[0][gs_ind],grids[1][gs_ind],grids[2][gs_ind])).T
print(gs_coord)
E_gs_raw = EE[gs_ind]
EE = EE - E_gs_raw
#########


#Define potential function
# note this interpolator only interpolates points or arrays of points, no grids.
V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential',symmExtend=False)


auxFunc = None # for MEP
#########
gs_coord = np.array((88,0,0))
E_gs = V_func(gs_coord)

V_func_shift = V_func#utilities.shift_func(V_func,shift=-1.0*E_gs)
const_names = ['pairing']
const_comps = [0]
plane_names = ['Q20','Q30','E_HFB']
re_dict= {'pairing':uniq_coords[2]}
xx_s,yy_s,zz_s = PES.get_2dsubspace(const_names,const_comps,plane_names)
zz_s = zz_s - E_gs_raw

fig, ax = plt.subplots(1,1,figsize = (12, 10))

im = ax.contourf(xx_s,yy_s,zz_s.clip(0, 10),cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,10))             
cs = ax.contour(xx_s,yy_s,zz_s.clip(0,13)+.01,colors=['black'],levels=18)
ax.clabel(cs,inline=5,fontsize=10,colors="black")
ax.contour(xx_s,yy_s,zz_s.clip(0,10),colors=['black'],levels=[E_gs],linewidths=3) 

#ax.plot(final_path_MEP[:, 0], final_path_MEP[:, 1], '.-',ms=10,label='MEP',color='red')    
ax.set_ylabel('$Q_{30}$',size=20)
ax.set_xlabel('$Q_{20}$',size=20)
ax.set_xlim([83,250])
ax.set_ylim([0,30])
cbar = fig.colorbar(im)
plt.show()  
plt.clf()
