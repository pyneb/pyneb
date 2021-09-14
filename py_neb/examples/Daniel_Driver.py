import sys
import os
sys.path.append(os.path.expanduser("~/Research/ActionMinimization/"))

from py_neb.py_neb import *
from py_neb import py_neb

import matplotlib.pyplot as plt

import time
import datetime
import cProfile
import itertools
 
from scipy.interpolate import interp2d, RBFInterpolator, interpn
from scipy import interpolate
import scipy.interpolate
import sklearn.gaussian_process as gp

def standard_pes(xx,yy,zz,clipRange=(-5,30)):
    #TODO: pull some (cleaner) options from ML_Funcs_Class
    #Obviously not general - good luck plotting a 3D PES lol
    fig, ax = plt.subplots()
    if clipRange is None:
        clipRange = (zz.min()-0.2,zz.max()+0.2)
    #USE THIS COLORMAP FOR PESs - has minima in blue and maxima in red
    cf = ax.contourf(xx,yy,zz.clip(clipRange[0],clipRange[1]),\
                     cmap="Spectral_r",levels=np.linspace(clipRange[0],clipRange[1],25))
    plt.colorbar(cf,ax=ax)
    
    ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)")
    return fig, ax

def h5_get_keys(obj):
    #Taken from https://stackoverflow.com/a/59898173
    keys = (obj.name,)
    if isinstance(obj,h5py.Group):
        for key, value in obj.items():
            if isinstance(value,h5py.Group):
                #Is recursive here
                keys = keys + h5_get_keys(value)
            else:
                keys = keys + (value.name,)
    return keys

def read_from_h5(fName):
    datDictOut = {}
    attrDictOut = {}
    
    h5File = h5py.File(fName,"r")
    allDataSets = [key for key in h5_get_keys(h5File) if isinstance(h5File[key],h5py.Dataset)]
    for key in allDataSets:
        datDictOut[key.lstrip("/")] = np.array(h5File[key])
        
    #Does NOT pull out sub-attributes
    for attr in h5File.attrs:
        attrIn = np.array(h5File.attrs[attr])
        #Some data is a scalar, and so would otherwise be stored as a zero-dimensional
        #numpy array. That's just confusing.
        if attrIn.shape == ():
            attrIn = attrIn.reshape((1,))
        attrDictOut[attr.lstrip("/")] = attrIn
    
    h5File.close()
        
    return datDictOut, attrDictOut

def main():
    fIn = "..//..//PES/232U.h5"
    dsets, attrs = read_from_h5(fIn)
    
    coordStrs = ["Q20","Q30"]
    
    uniqueCoords = [np.unique(dsets[key]) for key in coordStrs]
    
    gridShape = [len(np.unique(dsets[key])) for key in coordStrs]
    
    coordMesh = tuple([dsets[key].reshape(gridShape) for key in coordStrs])
    zz = dsets["PES"].reshape(gridShape)
    
    # potential = auxiliary_potential(NDInterpWithBoundary(uniqueCoords,zz),\
    #                                 shift=0.5)
    potential = auxiliary_potential(RectBivariateSplineWrapper(*uniqueCoords,zz).function)
    
    #Finding initial path
    gsLoc = np.array([attrs["Ground_State"][key] for key in coordStrs]).flatten()
    eGS = potential(gsLoc)
    
    nPts = 32
    initPath = np.array((np.linspace(gsLoc[0],300,nPts),\
                         np.linspace(gsLoc[1],32,nPts))).T
    
    f, a = standard_pes(*coordMesh,zz)
    a.contour(*coordMesh,zz,levels=[eGS],colors=["black"])
    
    lap = py_neb.LeastActionPath(potential,nPts,2,\
                                 nebParams={"k":10,"kappa":15},\
                                 endpointSpringForce=(False,True),\
                                 endpointHarmonicForce=(False,True))
    
    maxIters = 300
    tStep = 0.5
    
    minObj = py_neb.VerletMinimization(lap,initPath)
    tStepArr, alphaArr, _ = minObj.fire(tStep,maxIters,useLocal=True)
    
    minObj2 = py_neb.VerletMinimization(lap,initPath)
    tStepArr, alphaArr, _ = minObj2.fire(tStep,maxIters)
    
    # allPts, allVelocities, allForces = \
    #     minObj.velocity_verlet(tStep,maxIters)
    a.plot(minObj.allPts[-1,:,0],minObj.allPts[-1,:,1],marker=".",label="Local FIRE")
    a.plot(minObj2.allPts[-1,:,0],minObj2.allPts[-1,:,1],marker=".",label="Global FIRE")
    a.legend()
    # print("Slow interpolator time: "+str(t1 - t0))
    # print("Slow interpolator action: "+str(actions[-1]))
    
    actions = np.array([py_neb.action(minObj.allPts[i],potential)[0] for i in range(maxIters)])
    actions2 = np.array([py_neb.action(minObj2.allPts[i],potential)[0] for i in range(maxIters)])
    # print(actions)
    actionFig, actionAx = plt.subplots()
    actionAx.plot(actions,label="Local FIRE")
    actionAx.plot(actions2,label="Global FIRE")
    
    actionAx.legend()
    
    print(potential(minObj.allPts[-1]))
    # f2, a2 = plt.subplots()
    # a2.plot(tStepArr)
    
    return None

main()