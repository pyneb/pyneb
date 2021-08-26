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





if __name__ == "__main__":
    ### Define surface here
    data_path = '../PES/240Pu.h5'
    data = h5py.File(data_path, 'r')
    
    wanted_keys = ['Q20','Q30','pairing','E_HFB', 'B2020','B2030','B20pair','B3030','B30pair','Bpairpair']
    coord_keys = ['Q20','Q30','pairing'] ## warning, this has implict ordering for the grids.
    ## should contain all of the tensor components
    mass_keys = ['B2020','B2030','B20pair','B2030','B3030','B30pair','B20pair','B30pair','Bpairpair']
    
    data_dict = {}
    for key in wanted_keys:
        data_dict[key] = np.array(data[key])
    
    Q20 = np.sort(np.unique(data_dict['Q20']))
    Q30 = np.sort(np.unique(data_dict['Q30']))
    pairing = np.sort(np.unique(data_dict['pairing']))
    
    uniq_coord = [Q20,Q30,pairing]
    V = data_dict['E_HFB']
    
    
    l_bndy = np.zeros(len(uniq_coord))
    u_bndy = np.zeros(len(uniq_coord))
    for i,key in enumerate(coord_keys):
        l_bndy[i] = min(data_dict[key])
        u_bndy[i] = max(data_dict[key])
        
    coord_grids,EE = NEB.make_nd_grid(data,coord_keys,'E_HFB',return_grid=True)
    
    minima_ind = NEB.find_local_minimum(EE)
    ### ignore the fill values. Stolen from Dan's Code
    allowedInds = tuple(np.array([inds[EE[minima_ind]!=-1760] for inds in minima_ind]))
    gs_ind = NEB.extract_gs_inds(allowedInds,coord_grids,EE,pesPerc=0.5)
    E_gs = EE[gs_ind]
        
    gs_coord = np.array([coord_grids[i][gs_ind] for i in range(len(coord_grids))])
    EE = EE - E_gs
    # define potential function
    f = NEB.coord_interp_wrapper(uniq_coord,EE,l_bndy,u_bndy)
    # define mass function
    mass_tensor = NEB.mass_tensor_wrapper(data_dict,3,coord_keys,mass_keys,mass_func =None)

    N = 22
    M = 300
    dt = .1
    eta = 1.0 ## damping coeff for QMV

    ### params for BFGS

    alpha = 1.0
    beta = 1.0
    gamma = 0.5
    s_max = .1

    R0 = np.array(gs_coord) # start at GS
    RN = np.array((181,18,2)) # end at final OTL
    E_const = 0.0 ## constraint should be on the GS which is at 0 after shifting
    band =  NEB.NEB(f,mass_tensor,M,N,R0,RN,E_const,l_bndy,u_bndy)
    init_path = band.get_init_path()
    end_point_energy = band.get_end_points()
    k = 1.0
    kappa = 10.0
    fix_r0= True
    fix_rn=False
    #
    E_const = 0.0 ### constrain it to the ground state energy (assumed to be the starting point)
    force_params= {'E_const':E_const,'k':k,'kappa':kappa,'fix_r0':fix_r0,'fix_rn':fix_rn}
    plot_params = {'M':M,'N':N,'k':k,'E_gs':E_const}
    path_FIRE,action_FIRE,total_time_FIRE = band.FIRE(init_path,dt,eta,force_params,target='LAP')
    print(path_FIRE)
    proj_path = path_FIRE[:,[0,1]]
    proj_init = init_path[:,[0,1]]
    #path_QMV,action_QMV,total_time_QMV = band.QMV(init_path,dt,eta,force_params,target='LAP')

    #print(f((path_FIRE[-1][0],path_FIRE[-1][1])))
    plt.plot(np.arange(len(action_FIRE)),action_FIRE)
    #plt.plot(np.arange(len(action_QMV)),action_QMV)
    plt.show()
    
    ### slicing 
    const_names = ['pairing']
    const_comps = [0]
    plane_names = ['Q20','Q30','E_HFB']
    xx_s,yy_s,zz_s = NEB.subspace_2d(data_dict,const_names,const_comps,plane_names)


    zz_s = zz_s - E_gs
    #all_ccp = find_approximate_contours((xx,yy),zz,eneg=0,show=False)
    #otl = max(all_ccp[0], key=len)
    dan_path = np.loadtxt('./dan_demos/240Pu_Pairing_Final_Path.csv',skiprows=1,delimiter=',')
    dan_proj = dan_path[:,[0,1]]
    NEB.make_cplot([proj_init],[proj_path,dan_proj],[xx_s,yy_s],zz_s,plot_params,savefig=False)