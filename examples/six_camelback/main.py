import sys, os
pynebDir = '..//../src/'
if pynebDir not in sys.path:
    sys.path.insert(0,pynebDir)
import pyneb
 
import numpy as np
import matplotlib.pyplot as plt
import time

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
    
if __name__ == "__main__":
    os.makedirs('logs',exist_ok=True)
    
    """Setting up camelback potential on a grid"""
    x = np.arange(-2,2,0.05)
    y = np.arange(-1.25,1.25,0.05)
    coords = [x,y]
    
    xx, yy = np.meshgrid(x,y)
    zz = camelback(np.swapaxes(np.array([xx,yy]),0,-1))
    
    """Running NEB to compute the least action path"""
    nPts = 52
    nDims = 2
    
    loggerSettings = {"logName":"logs/camelback"}
    lap = pyneb.LeastActionPath(camelback,nPts,nDims,endpointSpringForce=False,
                                target_func=pyneb.TargetFunctions.action_squared,
                                target_func_grad=pyneb.GradientApproximations().discrete_sqr_action_grad,
                                endpointHarmonicForce=False,loggerSettings=loggerSettings)
    
    initialPath = np.array([np.linspace(-1.7,1.7,nPts),np.linspace(0.79,-0.79,nPts)]).T
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    tStep = 0.05
    nIters = 300
    t0 = time.time()
    nebObj.velocity_verlet(tStep,nIters)
    t1 = time.time()
    print('Finished running NEB in %.3f s'%(t1-t0))
    
    """Running dynamic programming to compute the least action path"""
    initialPoint = np.array([-1.7,0.8])
    allowedEndpoint = np.array([1.7,-0.8])
    dpm = pyneb.DynamicProgramming(initialPoint,(xx,yy),zz,allowedEndpoints=allowedEndpoint,
                                   fName="logs/camelback")
    t0 = time.time()
    _, minPathDict, _ = dpm()    
    t1 = time.time()
    print('Finished running DPM in %.3f s'%(t1-t0))
    dpmPath = list(minPathDict.values())[0]
    
    """Plotting the results"""
    fig, ax = plt.subplots()
    cf = ax.contourf(xx,yy,zz.T,cmap="Spectral_r",extend="both",levels=45)
    plt.colorbar(cf,ax=ax)
    ax.plot(*nebObj.allPts[-1].T,marker=".",color="black")
    ax.plot(*dpmPath.T,marker=".",color="red")
    
    ax.set(xlabel="x",ylabel="y",title="Camelback Potential")
    
