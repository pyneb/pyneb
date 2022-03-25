from context import pyneb
 
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp
import itertools
    
def camelback(coords):
    """
    6-camelback potential, shifted so that the global minimum energy is 0

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    assert isinstance(coords,np.ndarray)
    assert coords.shape[-1] == 2
    
    ndims = coords.ndim
    x, y = coords[(ndims-1)*(slice(None),)+(0,)], coords[(ndims-1)*(slice(None),)+(1,)]
    
    return (4 - 2.1*(x**2) + (1/3) * (x**4))*(x**2) + x*y + 4*((y**2) - 1)*(y**2) + 1.0315488275145395
    
def asymm_camelback(coords):
    assert isinstance(coords,np.ndarray)
    assert coords.shape[-1] == 2
    
    ndims = coords.ndim
    y = coords[(ndims-1)*(slice(None),)+(1,)]
    
    return camelback(coords) + 0.5*y

def run(params):
    nDims = 2
    
    tStep = 0.1
    nIters = 1000
    
    fireParams = {"maxmove":np.array(2*[0.2])}
    
    nPts, k, optString = params
    
    loggerSettings = {"logName":optString+"_k-"+str(k)+"_nPts-"+str(nPts)}
    lap = pyneb.LeastActionPath(camelback,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,loggerSettings=loggerSettings,
                                nebParams={"k":k})
    
    initialPath = np.array([np.linspace(-1.7,1.7,nPts),np.linspace(0.79,-0.79,nPts)]).T
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    if optString == "verlet":
        nebObj.velocity_verlet(tStep,nIters)
    elif optString == "local_fire":
        nebObj.fire(tStep,nIters,useLocal=True,earlyStop=False,fireParams=fireParams)
    elif optString == "global_fire":
        nebObj.fire(tStep,nIters,useLocal=False,earlyStop=False,fireParams=fireParams)
    else:
        nebObj.fire2(tStep,nIters,useLocal=False,earlyStop=False,fireParams=fireParams)
    
    # plt.plot([pyneb.TargetFunctions.action(p,camelback)[0] for p in nebObj.allPts])
    
    return None

if __name__ == "__main__":    
    nPtsArr = np.arange(20,210,20)
    kVals = np.array([0.01,0.05,0.1,0.5,1,5,10])
    optStrings = ["verlet","local_fire","global_fire","global_fire2"]
    
    paramsList = list(itertools.product(nPtsArr,kVals,optStrings))
    
    pool = mp.Pool()
    res = pool.map(run,paramsList)
