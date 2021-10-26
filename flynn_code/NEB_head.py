import NEB_modules as NEB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
from scipy.ndimage import filters, morphology #For minimum finding
import time
from matplotlib.pyplot import cm
import pandas as pd
import h5py
import multiprocessing as mp
from datetime import date
if __name__ == "__main__":
    today = date.today()
    #p = mp.Pool(mp.cpu_count())
    ### Define surface here
    nucleus = "232U"
    data_path = f"../PES/{nucleus}.h5"
    data = h5py.File(data_path, 'r')

    #wanted_keys = ['Q20','Q30','PES'] # keys for 240Pu
    wanted_keys = ['PES','Q20','Q30']
    #coord_keys = ['Q20','Q30'] ## coords for 252U. warning, this has implict ordering for the grids.
    
    coord_keys = ['Q20','Q30'] ## coords for 240Pu. warning, this has implict ordering for the grids.
    ## should contain all of the tensor components
    #mass_keys = ['B2020','B2030','B2030','B3030'] 
    mass_keys = []
    data_dict = {}
    for key in wanted_keys:
        data_dict[key] = np.array(data[key])
    
    Q20 = np.sort(np.unique(data_dict['Q20']))
    Q30 = np.sort(np.unique(data_dict['Q30']))
    #pairing = np.sort(np.unique(data_dict['pairing']))

    #uniq_coord = [Q20,Q30] #uniq_coord for 252U
    uniq_coord = [Q20,Q30] #uniq_coord for 240Pu
    
    #V = data_dict['E_HFB'] ## V for 240Pu
    V = data_dict['PES']
    
    dims = len(coord_keys)
    
    l_bndy = np.zeros(len(uniq_coord))
    u_bndy = np.zeros(len(uniq_coord))
    for i in np.arange(0,len(uniq_coord),1):
        l_bndy[i] = min(uniq_coord[i])
        u_bndy[i] = max(uniq_coord[i])
    coord_grids,EE = NEB.make_nd_grid(data,coord_keys,'PES',return_grid=True)

           
    minima_ind = NEB.find_local_minimum(EE)
    ### ignore the fill values. Stolen from Dan's Code
    allowedInds = tuple(np.array([inds[EE[minima_ind]!=-1760] for inds in minima_ind]))

    gs_ind = NEB.extract_gs_inds(allowedInds,coord_grids,EE,pesPerc=0.25)
    
    E_gs_shift = EE[gs_ind] 
    EE = EE - E_gs_shift
    gs_coord = np.array([coord_grids[i][gs_ind] for i in range(len(coord_grids))])
    

    
    # define potential function
    interp_kind = 'bivariant'
    mass_setting = None
    title = f"{nucleus}_{interp_kind}_Mass_{mass_setting}_Experimental"
    
    #f = NEB.coord_interp_wrapper(uniq_coord,EE,l_bndy,u_bndy)
    f = NEB.interp_wrapper(uniq_coord,EE,kind=interp_kind)
    #zz = f((coord_grids[0],coord_grids[1])) ### for ND linear interpolation
    
    zz = f((Q20,Q30)) ### for bivariant interpolation
    minima_ind = NEB.find_local_minimum(zz)
    local_minima = zz[minima_ind]
    order = np.argsort(local_minima)
    ordered_minima = local_minima[order]
    x_minima,y_minima = coord_grids[0][minima_ind],coord_grids[1][minima_ind]
    x_minima,y_minima = x_minima[order],y_minima[order]
    glb_min_idx = np.argmin(local_minima)
    glb_min = local_minima[glb_min_idx]
    glb_min_coords = (x_minima[glb_min_idx],y_minima[glb_min_idx])
    allowedInds = tuple(np.array([inds[zz[minima_ind]!=-1760] for inds in minima_ind]))

    gs_ind = NEB.extract_gs_inds(allowedInds,coord_grids,zz,pesPerc=0.25)
    E_gs = zz[gs_ind]
    

    # define mass function
    mass_tensor = NEB.mass_tensor_wrapper(data_dict,dims,coord_keys,mass_keys,mass_func=mass_setting,interp_kind=interp_kind)

    N = 32
    M = 600
    dt = .1
    eta = 1.0 ## damping coeff for QMV
    k = 10.0
    kappa = 20.0
    fix_r0= False
    fix_rn= False
    mu = 1.0
    R0 = np.array(gs_coord) # start at GS
    #RN = np.array((213.92,19.83)) # end at final OTL
    RN= np.array((280,22))
    E_const = E_gs ### constrain it to the ground state energy (assumed to be the starting point)
    band =  NEB.NEB(f,mass_tensor,M,N,R0,RN,E_const,l_bndy,u_bndy,)
    init_path = band.get_init_path()
    end_point_energy = band.get_end_points()
    
    
    
    force_params= {'E_const':E_const,'k':k,'mu':mu,'kappa':kappa,'fix_r0':fix_r0,'fix_rn':fix_rn}
    plot_params = {'M':M,'N':N,'k':k,'E_gs':E_const,'file_name':title }
    
    
    
    path_FIRE,action_FIRE,total_time_FIRE = band.FIRE(init_path,dt,eta,force_params,target='LAP2')

    #path_QMV,action_QMV,total_time_QMV = band.QMV(init_path,dt,eta,force_params,target='LAP2')
    final_action = np.around(action_FIRE[-1],2)
    #eric_action = np.around(action_QMV[-1],2)
    plt.plot(np.arange(len(action_FIRE)),action_FIRE,label='Eric Min '+str(final_action))
    #plt.plot(np.arange(len(action_QMV)),action_QMV,label='Eric Min '+str(eric_action))
    plt.xlabel('Iteration')
    plt.ylabel('Action')
    plt.legend()
    plt.title(title)
    plt.savefig(title+'_Action.pdf')
    plt.show()
    plt.clf()
    
    ### slicing for 240PU
    #const_names = ['pairing']
    #const_comps = [0]
    #plane_names = ['Q20','Q30','E_HFB']
    #xx_s,yy_s,zz_s = NEB.subspace_2d(data_dict,const_names,const_comps,plane_names)

    #zz_s = zz_s - E_gs
    


    method_dict = {'k':k,'kappa':kappa,'NImages': N,'Iterations':M,'optimization':'FIRE','fix_r0':fix_r0, \
                   'fix_rn': fix_rn}
    metadata = {'title':title,'Created_by': 'Eric','Created_on':today.strftime("%b-%d-%Y"),'method':'NEB','method_description':method_dict, \
                'surface_shift': str(E_gs_shift) +' and '+str(glb_min),'action':final_action,'run_time':total_time_FIRE,\
                    'initial_start_point': R0,'initial_end_point': RN}
    NEB.make_metadata(metadata)
    np.savetxt(title+'_path.txt',path_FIRE,comments='',delimiter=',',header="Q20\tQ30")
    print(total_time_FIRE)
    dan_path = np.loadtxt('../Paths/252U/252U_ND_Linear.txt',delimiter=',',skiprows= 1)
    names = [interp_kind,'dan_path']
    NEB.make_cplot([init_path],[path_FIRE,dan_path],[coord_grids[0],coord_grids[1]],zz,plot_params,names,savefig=True)
    