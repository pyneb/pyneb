import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
from matplotlib.pyplot import cm

from datetime import date
import sys
import pandas as pd
### add pyneb
sys.path.insert(0, '../../../py_neb/py_neb')
sys.path.insert(0, '../')
import solvers
import utilities
import utils
today = date.today()
### Define nucleus data path (assumes our github structure)
nucleus = "232U"
use_mass = True
save_data = False
save_plt = True
data_path = f"../../../PES/{nucleus}.h5"
### defines PES object from utils.py
PES = utils.PES(data_path)
mass_PES = utils.PES(data_path)
uniq_coords = PES.get_unique(return_type='array')
grids,EE = PES.get_grids(return_coord_grid=True,shift_GS=False)
mass_grids = PES.get_mass_grids()
mass_keys = mass_grids.keys()
### IMPORTANT: LIST THE INDICIES OF THE MASS TENSOR TO USE.
mass_tensor_indicies = ['20','30']

#Define potential function
# note this interpolator only interpolates points or arrays of points, no grids.
##
### the Edgar mass
m_e = .001
V_func = utilities.NDInterpWithBoundary(uniq_coords,np.sqrt(EE**2 +m_e**2),boundaryHandler='exponential',symmExtend=[True,True])


# Create dictionary of functions for each comp of the tensor
if use_mass == True:
    mass_grids = {key: utilities.NDInterpWithBoundary(uniq_coords,mass_grids[key],boundaryHandler='exponential',symmExtend=[True,True]) \
                  for key in mass_keys}
# function returns matrix of functions
    M_func = utilities.mass_funcs_to_array_func(mass_grids,mass_tensor_indicies)
else:
    M_func = None # for LAP 
auxFunc = None # for MEP



minima_ind = utilities.SurfaceUtils.find_all_local_minimum(EE)
gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.25,0.25],returnOnlySmallest=True)
gs_coord = np.array((grids[0][gs_ind],grids[1][gs_ind])).T
E_gs = EE[gs_ind]
#########
E_gs = V_func(gs_coord)

V_func_shift = utilities.shift_func(V_func,shift=E_gs)
NImgs = 82
k = 12.0
kappa = 1.0
E_const = E_gs
nDims = len(uniq_coords)
force_R0 = False
force_RN = False
springR0 = False
springRN = False
endPointFix = (force_R0,force_RN)
springForceFix = (springR0,springRN)
### Optimization parameters 
## Velocity Verlet parameter set
contours = utilities.SurfaceUtils.find_approximate_contours(grids,EE,eneg=0,show=False)
otl = contours[0][1]
otl_pnts = otl[100::40]
###  [298.49443073  30.78711465]
exit_point = np.array([[298.49, 30.79]])

otl_pnts = np.concatenate((otl_pnts,exit_point),axis=0)

for trial, RN in enumerate(otl_pnts):
    try:
        dt = .1
        NIterations = 50000
        ### define initial path
        R0 = gs_coord # NEB starting point
        #RN = [298.5,30.5] # NEB end point
        init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
        init_path = init_path_constructor.linear_path()
        
        ### Define parameter dictionaries (mostly for book keeping)
        neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}
        method_dict = {'k':k,'kappa':kappa,'NImages': NImgs,'Iterations':NIterations,'dt':dt,'optimization':'QM Verlet','HarmonicForceEnds': endPointFix, \
                           'SpringForceEnds': springForceFix}
        
        
        FireParams = \
            {"dtMax":5,"dtMin":0.001,"nAccel":10,"fInc":1.1,"fAlpha":0.99,\
             "fDecel":0.5,"aStart":0.1,"maxmove":[.25,.25]}
        
            
            
        #### Compute LAP
        # LAP function you want to minimize
        target_func_LAP = utilities.TargetFunctions.action_squared
        # LAP specialized function that takes the gradient of target function
        target_func_grad_LAP = utilities.GradientApproximations().discrete_sqr_action_grad_mp
        
        LAP_params = {'potential':V_func_shift,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                         'endpointHarmonicForce':endPointFix,'target_func':target_func_LAP,\
                         'target_func_grad':target_func_grad_LAP,'nebParams':neb_params}
        
        ### define the least action object 
        ### This essentially defines the forces given the target and gradient functions 
        lap = solvers.LeastActionPath(**LAP_params)
        
        ### Define the optimizer object to use. Note the initial band is passed
        ### here and the operations defined in LeastActionPath are applied to
        ### the band.
        minObj_LAP = solvers.VerletMinimization(lap,initialPoints=init_path)
        
        ### Begining the optimization procedure. Results are all of the velocities
        ### band positions, and forces for each iteration of the optimization.
        t0 = time.time()
        tStepArr, alphaArr, stepsSinceReset = minObj_LAP.fire(dt,NIterations,fireParams=FireParams,useLocal=False)
        allPaths_LAP = minObj_LAP.allPts
        #allPaths_LAP, allVelocities_LAP, allForces_LAP = minObj_LAP.velocity_verlet(dt,NIterations)
        final_path_LAP = allPaths_LAP[-1]
        t1 = time.time()
        total_time_LAP = t1 - t0
        print('total_time LAP: ',total_time_LAP)
        action_array_LAP = np.zeros(NIterations+2)
        for i,path in enumerate(allPaths_LAP):
            action_array_LAP[i] = utilities.TargetFunctions.action(path, V_func_shift,M_func)[0]
        min_action_LAP = np.around(action_array_LAP[-1],2)
        title = 'Eric_'+nucleus+'_LAP_Mass_'+str(use_mass)
        
        ### write run meta data to txt file.
        metadata = {'title':title,'Created_by': 'Eric','Created_on':today.strftime("%b-%d-%Y"),'method':'NEB-LAP','method_description':method_dict, \
                        'masses':None,'E_gs': str(E_gs),'action':action_array_LAP[-1],'run_time':total_time_LAP ,\
                            'initial_start_point': R0,'initial_end_point': RN}
        if save_data == True:
            utils.make_metadata(metadata)
        
        # write final path to txt.
        if save_data == True:
            np.savetxt(title+'_path'+str(trial)+'.txt',final_path_LAP,comments='',delimiter=',',header="Q20,Q30")
        
        print('end point: ', final_path_LAP[-1])
        
        
        
        ### Plot the results.
        fig, ax = plt.subplots(1,1,figsize = (12, 10))
        
        im = ax.contourf(grids[0],grids[1],EE,cmap='Spectral_r',extend='both',levels=MaxNLocator(nbins = 200).tick_values(0,15))
        ax.contour(grids[0],grids[1],EE,colors=['black'],levels=[E_gs])              
        ax.plot(init_path[:, 0], init_path[:, 1], '.-', color = 'orange',ms=10,label='Initial Path')
        ax.plot(final_path_LAP[:, 0], final_path_LAP[:, 1], '.-',ms=10,label='LAP',color='purple')
        ax.plot(otl_pnts[:,0],otl_pnts[:,1],marker='o',linestyle='',markersize=8,color='red')
        ax.plot(exit_point[0][0],exit_point[0][1],marker='*',linestyle='',markersize=12,color='yellow')
        ax.set_ylabel('$Q_{30}$',size=20)
        ax.set_xlabel('$Q_{20}$',size=20)
        ax.set_title('M = '+str(NIterations)+' N = '+str(NImgs)+' k='+str(k)+' kappa='+str(kappa)+'_mass_'+str(use_mass)+'_'+str(trial))
        
        
        ax.legend()
        cbar = fig.colorbar(im)
        if save_plt == True:
            plt.savefig(nucleus+'_Mass_'+str(use_mass)+'_M_'+str(NIterations)+'_N_'+str(NImgs)+'k_'+str(k)+'kappa_'+str(kappa)+'_path_'+str(trial)+'.pdf')
        plt.show()  
        plt.clf()
        
        plt.plot(range(NIterations+2),action_array_LAP,label='LAP '+str(min_action_LAP))
        #plt.plot(range(NIterations+2),action_array_MEP,label='MEP '+str(min_action_MEP))
        plt.xlabel('Iterations')
        plt.ylabel('Action')
        plt.legend()
        if save_plt == True:
            plt.savefig(nucleus+'_Mass_'+str(use_mass)+'_M_'+str(NIterations)+'_N_'+str(NImgs)+'k_'+str(k)+'kappa_'+str(kappa)+'_action_'+str(trial)+'.pdf')
        plt.show()
    except:
        raise('Band exploded for end point '+str(RN))