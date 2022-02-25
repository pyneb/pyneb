import numpy as np
import matplotlib.pyplot as plt
import utils
import sys
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, '../../py_neb/py_neb')
import utilities
nucleus = "240Pu"
data_path = f"../../PES/{nucleus}_new.h5"
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
gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.05,0.05,0.05],returnOnlySmallest=True)
gs_coord = np.array((grids[0][gs_ind],grids[1][gs_ind],grids[2][gs_ind])).T
print(gs_coord)
E_gs_raw = EE[gs_ind]
EE = EE - E_gs_raw
E_gs = EE[gs_ind]

V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,boundaryHandler='exponential')

const_names = ['pairing']
const_comps = [0]
plane_names = ['Q20','Q30','E_HFB']
re_dict= {'pairing':uniq_coords[2]}
xx_s,yy_s,zz_s = PES.get_2dsubspace(const_names,const_comps,plane_names)
zz_s = zz_s - E_gs_raw

LAP_nomass = np.loadtxt('../../Paths/240Pu/neb-lap_no_mass.txt',delimiter=',',skiprows=1)
MEP_nomass = np.loadtxt('../../Paths/240Pu/neb-mep_no_mass.txt',delimiter=',',skiprows=1)
DPM_nomass = np.loadtxt('../../Paths/240Pu/dpm_no_mass.txt',delimiter=',',skiprows=1)
ELE_nomass = np.loadtxt('../../Paths/240Pu/ele_no_mass.txt',delimiter=',',skiprows=1)
da_nomass = np.loadtxt('../../Paths/240Pu/dijkstra_no_mass.txt',delimiter=',',skiprows=1)
fig, ax = plt.subplots(1,1)
im = ax.contourf(xx_s,yy_s,zz_s.clip(0.01,10),cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 45).tick_values(0,10))
cs = ax.contour(xx_s,yy_s,zz_s.clip(0,4.5),colors=['black'],levels=np.arange(.5,10.5,.5),linewidths=.5) 
ax.contour(xx_s,yy_s,zz_s,colors=['white'],levels=[E_gs],linewidths=2.5)              
ax.clabel(cs,inline=1,fontsize=6,colors="black")

ax.plot(MEP_nomass[:,0], MEP_nomass[:,1],color='green')
#ax.plot(da_nomass[:,0], da_nomass[:,1],color='lime')
ax.plot(ELE_nomass[:,0], ELE_nomass[:,1],color='purple')
ax.plot(DPM_nomass[:,0], DPM_nomass[:,1],color='orange')
ax.plot(LAP_nomass[:,0], LAP_nomass[:,1],color='blue')
ax.set_xlim([86,280])
ax.set_ylim([0,30])
ax.set_ylabel(r'$Q_{30}$ ($b^{3/2}$)')
ax.set_xlabel(r'$Q_{20}$ ($b$)')
cbar = fig.colorbar(im,format="%.1f")
plt.savefig(nucleus+'_2d.pdf')
plt.show()  
plt.clf()
