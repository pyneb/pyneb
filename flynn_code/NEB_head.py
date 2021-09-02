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
    data_path = '../PES/252U_PES.h5'
    data = h5py.File(data_path, 'r')

    #wanted_keys = ['Q20','Q30','pairing','E_HFB', 'B2020','B2030','B20pair','B3030','B30pair','Bpairpair'] # keys for 240Pu
    wanted_keys = ['PES','Q20','Q30']
    coord_keys = ['Q20','Q30'] ## coords for 252U. warning, this has implict ordering for the grids.
    
    #coord_keys = ['Q20','Q30','pairing'] ## coords for 240Pu. warning, this has implict ordering for the grids.
    ## should contain all of the tensor components
    #mass_keys = ['B2020','B2030','B20pair','B2030','B3030','B30pair','B20pair','B30pair','Bpairpair'] #mass keys for 240Pu
    mass_keys = []
    data_dict = {}
    for key in wanted_keys:
        data_dict[key] = np.array(data[key])
    
    Q20 = np.sort(np.unique(data_dict['Q20']))
    Q30 = np.sort(np.unique(data_dict['Q30']))
    #pairing = np.sort(np.unique(data_dict['pairing']))

    uniq_coord = [Q20,Q30] #uniq_coord for 252U
    #uniq_coord = [Q20,Q30,pairing] #uniq_coord for 240Pu
    
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
    
    local_minima = EE[minima_ind]
    order = np.argsort(local_minima)
    ordered_minima = local_minima[order]
    x_minima,y_minima = coord_grids[0][minima_ind],coord_grids[1][minima_ind]
    x_minima,y_minima = x_minima[order],y_minima[order]
    glb_min_idx = np.argmin(local_minima)
    glb_min = local_minima[glb_min_idx]
    glb_min_coords = (x_minima[glb_min_idx],y_minima[glb_min_idx])
    print(ordered_minima)
    third_min_E = ordered_minima[1]
    third_min_coord = (x_minima[1],y_minima[1])
    EE = EE - glb_min
    gs_coord = np.array([coord_grids[i][gs_ind] for i in range(len(coord_grids))])
    E_gs = EE[gs_ind] 
    print(E_gs)
    
    # define potential function

    #f = NEB.coord_interp_wrapper(uniq_coord,EE,l_bndy,u_bndy)
    f = NEB.interp_wrapper(uniq_coord,EE,kind='bivariant')
    #print(f([ 33.80845336,1.72618612]))
    #zz = f((coord_grids[0],coord_grids[1])) ### for ND linear interpolation
    zz = f((Q20,Q30)) ### for bivariant interpolation
    
    
    
    # define mass function
    mass_tensor = NEB.mass_tensor_wrapper(data_dict,dims,coord_keys,mass_keys,mass_func = None)
    N = 52
    M = 3
    dt = .1
    eta = 1.0 ## damping coeff for QMV
    k = 10.0
    kappa = 20.0
    fix_r0= False
    fix_rn= False
    mu = 1.0
    
    R0 = np.array(gs_coord) # start at GS
    RN = np.array((280,15)) # end at final OTL
    E_const = E_gs ## constraint should be on the GS which is at 0 after shifting

    band =  NEB.NEB(f,mass_tensor,M,N,R0,RN,E_const,l_bndy,u_bndy)
    init_path = band.get_init_path()
    end_point_energy = band.get_end_points()
    
    
    E_const = E_gs ### constrain it to the ground state energy (assumed to be the starting point)
    force_params= {'E_const':E_const,'k':k,'mu':mu,'kappa':kappa,'fix_r0':fix_r0,'fix_rn':fix_rn}
    plot_params = {'M':M,'N':N,'k':k,'E_gs':E_const}
    #path_FIRE,action_FIRE,total_time_FIRE = band.FIRE(init_path,dt,eta,force_params,target='LAP')

    path_QMV,action_QMV,total_time_QMV = band.QMV(init_path,dt,eta,force_params,target='LAP2')
    #eric_action = np.around(action_FIRE[-1],2)
    eric_action = np.around(action_QMV[-1],2)
    #plt.plot(np.arange(len(action_FIRE)),action_FIRE,label='Eric Min '+str(eric_action))
    plt.plot(np.arange(len(action_QMV)),action_QMV,label='Eric Min '+str(eric_action))
    plt.xlabel('Iteration')
    plt.ylabel('Action')
    plt.legend()
    plt.show()
    plt.clf()
    '''
    ### slicing for 240PU
    const_names = ['pairing']
    const_comps = [0]
    plane_names = ['Q20','Q30','E_HFB']
    xx_s,yy_s,zz_s = NEB.subspace_2d(data_dict,const_names,const_comps,plane_names)

    zz_s = zz_s - E_gs
    '''
    #np.savetxt('QMV_final_path_bivar_order5.txt',path_QMV)
    #np.savetxt('QMV_action_bivar_order5.txt',action_QMV)
    '''
    ND_lin_path = np.loadtxt('QMV_final_path_NDLinear.txt')
    order1 = np.loadtxt('QMV_final_path_bivar_order1.txt')
    order5 = np.loadtxt('QMV_final_path_bivar_order5.txt')
    dan1 = np.loadtxt('./dan_demos/paths/ND_Linear.txt',delimiter=',')
    dan2 = np.loadtxt('./dan_demos/paths/2D_Linear.txt',delimiter=',')
    dan3 = np.loadtxt('./dan_demos/paths/2D_Quintic.txt',delimiter=',')
    sly = np.loadtxt('../PES/fission_path_dpm_252U.txt')
    '''
    names = ['NDLinear','Bivariant Order 1','Bivariant Order 5','Dan NDLinear','Dan Order 1','Dan Order 5','Dynamic Programming']
    NEB.make_cplot([init_path],[path_QMV],[coord_grids[0],coord_grids[1]],EE,plot_params,names,savefig=False)

    