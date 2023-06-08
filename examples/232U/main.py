import sys
import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

import time

pyNebDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..//../src"))
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
import pyneb

def read_potential():
    dsetsToGet = ["Q20","Q30","PES","B2020","B2030","B3030"]
    dsetsDict = {}
    
    h5File = h5py.File("232U.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    return dsetsDict

if __name__ == '__main__':
    os.makedirs('logs',exist_ok=True)
    
    """General initialization"""
    action = pyneb.TargetFunctions.action
    allPathsDict = {}
    allActionsTrainDict = {}
    allFinalActionsDict = {}
    
    dsetsDictIn = read_potential()
    coords = ["Q20","Q30"]
    uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
    pesShape = [len(c) for c in uniqueCoords]
    
    dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
    
    coordMeshTuple = tuple([dsetsDict[c] for c in coords])
    zz = dsetsDict["PES"]
    
    #Shifting the ground state to be positive. Note that the PES is not always positive
    gsInds = pyneb.SurfaceUtils.find_local_minimum(zz)
    gsLoc = np.array([c[gsInds] for c in coordMeshTuple])
    zz -= zz[gsInds]
    
    #Setting up interpolators
    potential = pyneb.NDInterpWithBoundary(uniqueCoords,zz.T)
    inertiaFunc = \
        pyneb.PositiveSemidefInterpolator(uniqueCoords,
                                          [dsetsDict['B2020'],dsetsDict['B2030'],dsetsDict['B3030']])
    
    #Finding endpoints
    allowedEndpoints = pyneb.SurfaceUtils.find_endpoints_on_grid(coordMeshTuple,zz)
    
    start = gsLoc
    end = np.array([298.,31.2]) #Selected with prior knowledge of the PES to be near
                                #the outer turning line
    
    nPts = 30
    nDims = 2
    
    initialPath = \
        np.vstack([np.linspace(start[cIter],end[cIter],nPts) for cIter in range(nDims)]).T
    tStep = 0.25
    maxIters = 250
    useLocal = True
    
    tfGrad = pyneb.GradientApproximations().discrete_sqr_action_grad
    
    """Getting LAP with NEB"""
    lap = pyneb.LeastActionPath(potential,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,
                                target_func_grad=tfGrad,
                                loggerSettings={"logName":"logs/no_inertia"})
    
    verletLAP = pyneb.VerletMinimization(lap,initialPath)
    t0 = time.time()
    verletLAP.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
    t1 = time.time()
    print('Finished NEB-LAP with id inertia in %.3f s'%(t1-t0))
    verletLAPAction = np.array([action(path,potential)[0] for path in verletLAP.allPts])
    
    """Getting LAP with inertia using NEB"""
    lapInertia = pyneb.LeastActionPath(potential,nPts,nDims,mass=inertiaFunc,
                                       endpointSpringForce=False,endpointHarmonicForce=False,
                                       target_func_grad=tfGrad,
                                       loggerSettings={"logName":"logs/inertia"})
    
    verletLAPInertia = pyneb.VerletMinimization(lapInertia,initialPath)
    t0 = time.time()
    verletLAPInertia.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
    t1 = time.time()
    print('Finished NEB-LAP with variable inertia in %.3f s'%(t1-t0))
    verletLAPActionInertia = np.array([action(path,potential,masses=inertiaFunc)[0] for \
                                       path in verletLAPInertia.allPts])
        
    """Getting minimum energy path with NEB"""
    mep = pyneb.MinimumEnergyPath(potential,nPts,nDims,endpointSpringForce=False,
                                  endpointHarmonicForce=False,
                                  loggerSettings={"logName":"logs/no_inertia"})
    
    verletMEP = pyneb.VerletMinimization(mep,initialPath)
    t0 = time.time()
    verletMEP.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
    t1 = time.time()
    print('Finished NEB-MEP in %.3f s'%(t1-t0))
    verletMEPAction = np.array([action(path,potential)[0] for path in verletMEP.allPts])
        
    """Plotting results"""
    fig, ax = plt.subplots()
    cf = ax.contourf(*coordMeshTuple,zz.clip(-5,50),levels=np.arange(-5,51,1),cmap="Spectral_r")
    ax.contour(*coordMeshTuple,zz,levels=[0],colors=["white"])
    plt.colorbar(cf,ax=ax)
    
    ax.scatter(*start,color="red",marker="x")
    ax.scatter(*end,color="red",marker="^")
    
    ax.set(xlabel="Q20",ylabel="Q30",title=r"${}^{232}$U")
    
    ax.plot(verletLAP.allPts[-1,:,0],verletLAP.allPts[-1,:,1],color="blue",\
            label="LAP (%.3f)" % verletLAPAction[-1])
    ax.plot(verletMEP.allPts[-1,:,0],verletMEP.allPts[-1,:,1],color="green",\
            label="MEP (%.3f)" % verletMEPAction[-1])
        
    ax.plot(verletLAPInertia.allPts[-1,:,0],verletLAPInertia.allPts[-1,:,1],"--",color="blue",\
            label="LAP (%.3f)" % verletLAPActionInertia[-1])
    
    ax.legend()
    
    """Plotting convergence"""
    #The minimum action is subtracted, to highlight the convergence
    fig, ax = plt.subplots()
    ax.plot(verletLAPAction - verletLAPAction.min(),label="LAP No Inertia",color="blue")
    ax.plot(verletMEPAction - verletMEPAction.min(),label="MEP",color="green")
    ax.plot(verletLAPActionInertia - verletLAPActionInertia.min(),color="blue",ls="--")
    
    ax.legend()
    ax.set(xlabel="Iterations",ylabel="Action",title=r"$S-S_{\mathrm{min}}$ vs Iteration Number",
           ylim=(0,20))
