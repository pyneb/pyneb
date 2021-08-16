import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
from scipy.ndimage import filters, morphology #For minimum finding
from scipy.signal import argrelextrema
import time
import cProfile

from NEB_Class import *

import os, h5py
global logDict, outputNms
logDict = {}
outputNms = {}
    
def update_log(strRep,outputTuple,outputNmsTuple,isTuple=True):
    global logDict, outputNms
    
    #If returning a single value, set isTuple -> False 
    if not isTuple:
        outputTuple = (outputTuple,)
        outputNmsTuple = (outputNmsTuple,)
    
    if strRep not in logDict:
        logDict[strRep] = []
        for t in outputTuple:
            if isinstance(t,np.ndarray):
                logDict[strRep].append(np.expand_dims(t,axis=0))
            else:
                logDict[strRep].append([t])
        outputNms[strRep] = outputNmsTuple
    else:
        assert len(outputTuple) == len(logDict[strRep])
        for (tIter,t) in enumerate(outputTuple):
            if isinstance(t,np.ndarray):
                logDict[strRep][tIter] = \
                    np.concatenate((logDict[strRep][tIter],np.expand_dims(t,axis=0)))
            else:
                logDict[strRep][tIter].append(t)
                    
    return None
    
def write_log(fName,overwrite=False):
    #WARNING: probably doesn't handle anything that isn't a numpy array, although
        #that's almost all that I intend to log at the moment
    global logDict, outputNms

    if not fName.startswith("Logs/"):
        fName = "Logs/"+fName
    os.makedirs("Logs",exist_ok=True)
    
    if (overwrite) and (os.path.isfile(fName)):
        os.remove(fName)
    
    h5File = h5py.File(fName,"a")
    for key in logDict.keys():
        splitKey = key.split(".")
        for (sIter,s) in enumerate(splitKey):
            subGp = "/".join(splitKey[:sIter+1])
            if not subGp in h5File:
                h5File.create_group(subGp)
        for (oIter,outputNm) in enumerate(outputNms[key]):
            h5File[key.replace(".","/")].create_dataset(outputNm,data=logDict[key][oIter])
    
    h5File.close()
    
    return None

def find_local_minimum(arr):
    """
    Returns the indices corresponding to the local minimum values. Taken 
    directly from https://stackoverflow.com/a/3986876
    
    Parameters
    ----------
    arr : Numpy array
        A D-dimensional array.

    Returns
    -------
    minIndsOut : Tuple of numpy arrays
        D arrays of length k, for k minima found

    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood,\
                                        mode="nearest")==arr)
    
    background = (arr==0)
    #Not sure this is necessary - it doesn't seem to do much on the test
        #data I defined.
    eroded_background = morphology.binary_erosion(background,\
                                                  structure=neighborhood,\
                                                  border_value=1)
        
    detected_minima = local_min ^ eroded_background
    allMinInds = np.vstack(local_min.nonzero())
    minIndsOut = tuple([allMinInds[coordIter,:] for \
                        coordIter in range(allMinInds.shape[0])])
    return minIndsOut

def V_Syl(x,y):
    A = -0.8447
    B = -0.2236
    C = 0.1247
    D = -4.468
    E = 0.02194
    F = 0.3041
    G = 0.1687
    H = 0.4388
    I = -4.713 * 10**(-7)
    J = -1.148 * 10**(-5)
    K = 1.687
    L = -3.062 * 10**(-18)
    M = -9.426 * 10**(-6)
    N = -2.851 * 10**(-16)
    O = 2.313 * 10**(-5)
    
    vOut = A + B*x + C*y + D*x**2 + E*x*y + F*y**2 + G*x**3 + H*x**2*y
    vOut += I*x*y**2 + J*y**3 + K*x**4 + L*x**3*y + M*x**2*y**2 + N*x*y**3 + O*y**4
    return vOut

def action(path,V,minimum):
    enegs = V(*path.T)
    sqrtTerm = np.sqrt(2*(enegs-minimum))
    
    a = 0
    for i in range(1,path.shape[0]):
        dist = np.linalg.norm(path[i]-path[i-1])
        a += (sqrtTerm[i] + sqrtTerm[i-1]) * dist
    
    return a/2

def grad_2d(f,x,y):
    strRep = "grad_2d"
    
    h = 10**(-8)
    
    df_dx = (f(x+h,y)-f(x,y))/h
    df_dy = (f(x,y+h)-f(x,y))/h
    
    ret = (df_dx,df_dy)
    nmsTuple = ("df_dx","df_dy")
    update_log(strRep,ret,nmsTuple)
    
    return ret

def eps(V,x,y,m,E_gs):
    result = np.sqrt(2*m*(V(x,y) - E_gs))
    return(result)

def get_tang_vect(int_path,V):
    strRep = "get_tang_vect"
    
    #returns array of tangen vectors for each point on the chain. The position of each is given by chain coords. 
    #each row is the ith tangent vector directional information. 
    tan_vects = np.zeros(int_path.shape)
    for i in np.arange(0,len(int_path),1):
        if i==0:
            tan = (int_path[i+1] - int_path[i])
        elif i == len(int_path)-1:
            tan = (int_path[i] - int_path[i-1])
        else:
            Vip1 = V(int_path[i+1][0],int_path[i+1][1])
            Vi = V(int_path[i][0],int_path[i][1])
            Vim1 =V(int_path[i-1][0],int_path[i-1][1])
            if (Vip1 > Vi) and (Vi > Vim1): 
                tan = int_path[i+1] - int_path[i]
            elif (Vip1 < Vi) and (Vi < Vim1): 
                tan = int_path[i] - int_path[i-1]
            elif (Vip1 > Vim1):
                delta_V_min = min(abs(Vip1 - Vi),abs(Vim1 - Vi))
                delta_V_max = max(abs(Vip1 - Vi),abs(Vim1 - Vi))
                tan = (int_path[i+1] - int_path[i])*delta_V_max + (int_path[i] - int_path[i-1])*delta_V_min
            else:
                delta_V_min = min(abs(Vip1 - Vi),abs(Vim1 - Vi))
                delta_V_max = max(abs(Vip1 - Vi),abs(Vim1 - Vi))
                tan = (int_path[i+1] - int_path[i])*delta_V_min + (int_path[i] - int_path[i-1])*delta_V_max
                
        norm = np.linalg.norm(tan)
        tan = tan/norm
        tan_vects[i] = tan
        
    ret = tan_vects
    nmsTuple = "tan_vects"
    update_log(strRep,ret,nmsTuple,isTuple=False)
        
    return ret

def F_s(k,R,tan_vects):
    strRep = "F_s"
    
    #WARNING: spring force is ALWAYS zero on the endpoints
    #returns 2d-array calculating force at each image.
    # R is an array of the position vectors on the chain. each ith row is assumed to be R_{i}
    force = np.zeros(R.shape)
    for i in range(1,len(R)-1):
        force[i] = k*(np.linalg.norm(R[i+1] - R[i]) - np.linalg.norm(R[i]  - R[i-1]))*tan_vects[i]
    
    ret = force
    nmsTuple = "force"
    update_log(strRep,ret,nmsTuple,isTuple=False)
    
    return ret

def negative_target_gradient(V,path,m,E):
    strRep = "negative_target_gradient"
    
    g = np.zeros(path.shape)
    
    for i in range(1,len(path)-1):
        f = -1*np.array(grad_2d(V,path[i][0],path[i][1]))
        d_i = np.linalg.norm(path[i] - path[i-1])
        d_ip1 = np.linalg.norm(path[i+1] - path[i])
        d_ivec = (path[i] - path[i-1])/d_i
        d_ip1vec = (path[i+1] - path[i])/d_ip1
        g[i] =.5*((m/E[i])*(d_i + d_ip1)*f - (E[i] + E[i-1])*d_ivec + (E[i+1] + E[i])*d_ip1vec) 
        
    ret = g
    nmsTuple = "g"
    update_log(strRep,ret,nmsTuple,isTuple=False)
    
    return ret

def g_perp(V,path,m,tau,E_gs,E_const,k,kappa,fix_r0,fix_rn):
    strRep = "g_perp"
    
    N_idx = np.arange(0,len(path),1)
    g_perp= np.zeros((len(N_idx),2))
    E = eps(V,path[:,0],path[:,1],m,E_gs)
    g = negative_target_gradient(V,path,m,E)
    for i in range(1,len(path)-1):
        g_perp[i] = g[i] - np.dot(g[i],tau[i])*tau[i]
        
    if fix_r0:
        g_perp[0] = np.zeros(2)
    else:
        g_spr_0 = k*(path[1]-path[0])
        f = -1*np.array(grad_2d(V,path[0][0],path[0][1]))
        f_norm = np.linalg.norm(f)
        g_perp[0] = g_spr_0 - (np.dot(g_spr_0,f/f_norm) - kappa*(V(path[0][0],path[0][1]) - E_const))*f/f_norm
        
    if fix_rn:
        g_perp[-1] = np.zeros(2)
    else:
        g_spr_0 = k*(path[-1]-path[-2])
        f = -1*np.array(grad_2d(V,path[-1][0],path[-1][1]))
        f_norm = np.linalg.norm(f)
        g_perp[-1] = g_spr_0 - (np.dot(g_spr_0,f/f_norm) - kappa*(V(path[-1][0],path[-1][1]) - E_const))*f/f_norm
        
    ret = g_perp
    nmsTuple = "g_perp"
    update_log(strRep,ret,nmsTuple,isTuple=False)
        
    return g_perp

def NEB_LAP(V,init_path,E_gs,E_const,M,N,dt,k,kappa,fix_r0=False,fix_rn=False):
    strRep = "NEB_LAP"
    
    action_array = np.zeros((M))
    # path = np.zeros((M,N,2))
    # path[0] = init_path
    path = np.full((M,N,2),init_path)
    v = np.full((M,N,2),np.zeros(init_path.shape))
    forces = np.zeros((M,2,N))
    
    #### MAIN KERNEL (QM Verlet)
    for i in np.arange(0,M-1,1):
        tau = get_tang_vect(path[i],V)
        F_spring = F_s(k,path[i],tau)
        g = g_perp(V,path[i],1.0,tau,E_gs,E_const,k,kappa,fix_r0,fix_rn)
        F =  F_spring + g
        forces[i] = F.T
        for j in np.arange(1,N-1,1):
            #Fixing off-by-one error in prod
            prod = np.dot(v[i,j],F[j])
            if prod > 0:
                vProj= prod*F[j]/np.linalg.norm(F[j])
            else:
                vProj = np.zeros(v[i,j].shape)
            v[i+1,j] = vProj + dt*F[j]
            path[i+1][j] = path[i][j] + v[i+1,j]*dt + .5*F[j]*dt**2
        action_array[i+1] = action(path[i+1],V,E_gs)
        
    action_array[0] = action(path[0],V,E_gs)
    
    ret = (path[-1],action_array,path,forces)
    nmsTuple = ("path[-1]","action_array","path","forces")
    update_log(strRep,ret,nmsTuple)
    
    return ret

def test_movable_endpoints():
    LAP_path,LAP_action_array,fullpath,forces =  \
        NEB_LAP(V,init_path,E_gs,E_const,M_LAP,N,dt_LAP,k,kappa)
    
    f, a = plt.subplots()
    a.plot(np.arange(0,M_LAP,1),LAP_action_array,label="No Fixed")
    a.axhline(LAP_action_array[-1],0,M_LAP,label='Converged Val = '+str(np.around(LAP_action_array[-1],3)),color='red')
    a.set(xlabel='Iteration',ylabel='Action')
    plt.legend()
    
    fig, ax = plt.subplots()
    im = ax.contour(rrAB,xx, zz, colors=['black'],levels=MaxNLocator(nbins = 50).tick_values(-5,12))                
    ax.plot(init_path[:,0], init_path[:, 1], '.-',label='Initial')
    ax.plot(LAP_path[:, 0], LAP_path[:, 1], '.-',label='No Fixed')
    
    LAP_path,LAP_action_array,fullpath,forces =  \
        NEB_LAP(V,init_path,E_gs,E_const,M_LAP,N,dt_LAP,k,kappa,fix_r0=True)
    
    a.plot(np.arange(0,M_LAP,1),LAP_action_array,label="R0 Fixed")
    a.axhline(LAP_action_array[-1],0,M_LAP,label='Converged Val = '+str(np.around(LAP_action_array[-1],3)),color='red')
    
    ax.plot(LAP_path[:, 0], LAP_path[:, 1], '.-',label='R0 Fixed')
    
    LAP_path,LAP_action_array,fullpath,forces =  \
        NEB_LAP(V,init_path,E_gs,E_const,M_LAP,N,dt_LAP,k,kappa,fix_rn=True)
    
    a.plot(np.arange(0,M_LAP,1),LAP_action_array,label="RN Fixed")
    a.axhline(LAP_action_array[-1],0,M_LAP,label='Converged Val = '+str(np.around(LAP_action_array[-1],3)),color='red')
    
    ax.plot(LAP_path[:, 0], LAP_path[:, 1], '.-',label='RN Fixed')
    
    LAP_path,LAP_action_array,fullpath,forces =  \
        NEB_LAP(V,init_path,E_gs,E_const,M_LAP,N,dt_LAP,k,kappa,fix_r0=True,fix_rn=True)
    
    a.plot(np.arange(0,M_LAP,1),LAP_action_array,label="Both Fixed")
    a.axhline(LAP_action_array[-1],0,M_LAP,label='Converged Val = '+str(np.around(LAP_action_array[-1],3)),color='red')
    
    ax.plot(LAP_path[:, 0], LAP_path[:, 1], '.-',label='Both Fixed')
    
    ax.legend()
    
    return None

def default_run():
    LAP_path,LAP_action_array,fullpath,forces =  \
        NEB_LAP(V,init_path,E_gs,E_const,M_LAP,N,dt_LAP,k,kappa)
    
    # f, a = plt.subplots()
    # a.plot(np.arange(0,M_LAP,1),LAP_action_array,label="No Fixed")
    # a.axhline(LAP_action_array[-1],0,M_LAP,label='Converged Val = '+str(np.around(LAP_action_array[-1],3)),color='red')
    # a.set(xlabel='Iteration',ylabel='Action')
    # plt.legend()
    
    fig, ax = plt.subplots()
    im = ax.contour(rrAB,xx, zz, colors=['black'],levels=MaxNLocator(nbins = 50).tick_values(-5,12))                
    ax.plot(init_path[:,0], init_path[:, 1], '.-',label='Initial')
    ax.plot(LAP_path[:, 0], LAP_path[:, 1], '.-',label='No Fixed')
    ax.legend()
    
    write_log("Eric_Log.h5",overwrite=True)
    FileIO.write_path("./Paths/Erics_Path.txt",LAP_path)
    
    return None

V=V_Syl

rrAB,xx = np.meshgrid(np.arange(-2,2,0.05),np.arange(-3,3,0.05))
zz = V(rrAB,xx)
minima = find_local_minimum(zz)

N = 22
M_LAP = 2500
dt_LAP = .1
k = 10.0
kappa = 1.0

init_path = np.array([np.linspace(c[minima][0],c[minima][1],num=N) for\
                      c in (rrAB,xx)]).T
E_gs, E_const = zz[minima]

default_run()
# cProfile.run("default_run()",sort="tottime")
