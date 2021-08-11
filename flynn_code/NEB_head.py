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
    data_path = '../252U_Test_Case/252U_PES.h5'
    data = h5py.File(data_path, 'r')
    Q20_grid = np.array(data['Q20'])
    Q30_grid = np.array(data['Q30'])
    V_grid = np.array(data['PES'])

    '''
    fig, ax = plt.subplots()
    ax.contourf(xx,yy,zz,levels=MaxNLocator(nbins = 200).tick_values(-2,20))
    ccp = ax.contour(xx,yy,zz,levels=[0])
    turning_line= max(ccp.allsegs[0], key=len)
    '''
    ### interpolate the grid
    nx = 500 # number of points on x-axis
    ny = 500 # number of points on y-axis
    Q20 = np.linspace(Q20_grid[:,0][0],Q20_grid[:,0][-1],nx)
    Q30 = np.linspace(Q30_grid[0][0],Q30_grid[0][-1],ny)

    xx, yy = np.meshgrid(Q20,Q30)

    ### create interpolation function
    f = interpolate.RectBivariateSpline(Q20_grid[:,1], Q30_grid[0], V_grid, kx=5, ky=5, s=0)

    zz = f(Q20,Q30).T
    minima_ind = NEB.find_local_minimum(zz)
    local_minima = zz[minima_ind]
    order = np.argsort(local_minima)
    ordered_minima = local_minima[order]
    x_minima,y_minima = xx[minima_ind],yy[minima_ind]
    x_minima,y_minima = x_minima[order],y_minima[order]
    glb_min_idx = np.argmin(local_minima)
    glb_min = local_minima[glb_min_idx]
    glb_min_coords = (x_minima[glb_min_idx],y_minima[glb_min_idx])

    N = 52
    M = 200
    dt = .1
    eta = 1.0 ## damping coeff for QMV

    x_lims = (Q20[0],Q20[-1])
    y_lims = (Q30[0],Q30[-1])
    grid_size = V_grid.shape
    ### params for BFGS
    '''
    alpha = 1.0
    beta = 1.0
    gamma = 0.5
    s_max = .1
    '''
    R0 = (25.95,0.96) # start at GS
    #RN = (213.92,19.83) # end at "third" minimum 
    RN = (281.96,25.31) # end at final OTL
    band =  NEB.NEB(f,M,N,x_lims,y_lims,grid_size,R0,RN,glb_min)
    init_path = band.get_init_path()
    minima = band.get_end_points()
    m = 1.0
    k = 3.0
    kappa = 1.0
    fix_r0=True
    fix_rn=False
    E_const = minima[0] ### constrain it to the ground state energy (assumed to be the starting point)
    force_params= {'E_const':E_const,'m':m,'k':k,'kappa':kappa,'fix_r0':fix_r0,'fix_rn':fix_rn}
    plot_params = {'M':M,'N':N,'k':k,'E_gs':E_const}
    path_FIRE,action_FIRE,energies_FIRE,total_time_FIRE = band.FIRE(init_path,dt,eta,force_params,target='LAP')
    #path_QMV,action_QMV,energies_QMV,total_time_QMV = band.QMV(init_path,dt,eta,force_params,target='LAP')


    print(f(path_FIRE[-1][0],path_FIRE[-1][1]).item() -glb_min)
    #plt.plot(np.arange(len(action_FIRE)),action_FIRE)
    #plt.plot(np.arange(len(action_QMV)),action_QMV)
    plt.show()
    '''
    ### second band
    R0 = path_QMV[-1] ## start second band at first band ending
    RN = (281.96,25.31) ## end at final otp
    ### TODO: add otl finder from ML code
    N = 6
    band2 = NEB.NEB(f,M,N,x_lims,y_lims,grid_size,R0,RN)
    init_path2 =  band2.get_init_path()
    path_FIRE2,action_array_FIRE2,energies_FIRE2,total_time_FIRE2 = band2.FIRE(init_path2,dt,eta,force_params,target='LAP')
    '''
    NEB.make_cplot([init_path],[path_FIRE],[xx,yy],zz-glb_min,plot_params,savefig=False)