import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys, os

pynebDir = '..//../src/'
if pynebDir not in sys.path:
    sys.path.insert(0,pynebDir)
import pyneb

def muller_brown(coords):
    assert isinstance(coords,np.ndarray)
    assert coords.shape[-1] == 2
    
    ndims = coords.ndim
    x, y = coords[(ndims-1)*(slice(None),)+(0,)], coords[(ndims-1)*(slice(None),)+(1,)]
    
    result = np.zeros(x.shape)
    for i in range(4):
        result += mbParams['A'][i]*np.exp(mbParams['a'][i]*(x-mbParams['x_bar'][i])**2 \
                                          + mbParams['b'][i]*(x-mbParams['x_bar'][i])*(y-mbParams['y_bar'][i]) \
                                              + mbParams['c'][i]*(y-mbParams['y_bar'][i])**2)
    return result

if __name__ == "__main__":
    os.makedirs('logs',exist_ok=True)
    
    mbParams = dict(A = [-200,-100,-170,15],
                    a = [-1,-1,-6.5,0.7],
                    b = [0,0,11,0.6],
                    c = [-10,-10,-6.5,0.7],
                    x_bar = [1,0,-0.5,-1],
                    y_bar = [0,0.5,1.5,1])
    
    """Setting up Muller-Brown potential, and subtracting ground state energy"""
    #Use unequal shapes to easily know shapes of arrays we're after
    x = np.linspace(-1.5, 1,300)
    y = np.linspace(-.25,2,290)
    xx, yy = np.meshgrid(x,y)
    zz = muller_brown(np.swapaxes(np.array([xx,yy]),0,-1))
    
    gsInds = pyneb.SurfaceUtils.find_local_minimum(zz,searchPerc=[1,1])
    gsLoc = np.array((xx.T[gsInds],yy.T[gsInds]))
    gsEneg = muller_brown(gsLoc)
    
    zz -= gsEneg
    
    pes = pyneb.shift_func(muller_brown,gsEneg)
    
    """Plotting the Muller-Brown function"""
    pesFig, pesAx = plt.subplots()
    cf = pesAx.contourf(xx,yy,zz.T.clip(max=200),cmap='Spectral_r',levels=30)
    plt.colorbar(cf,ax=pesAx)
    pesAx.scatter(*gsLoc,marker='x',color='black')
    
    pesAx.set(xlabel='x',ylabel='y')
    
    """NEB for the least action path"""
    #NEB parameters
    nPts = 32
    nDims = 2
    k = 3
    kappa = 1
    
    #Optimization parameters
    dt = 0.01
    nIters = 500
    
    initialPath = np.array([np.linspace(-0.55,0.62,nPts),np.linspace(1.44,0.03,nPts)]).T
    
    #Setting up
    lap = pyneb.LeastActionPath(pes,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,
                                nebParams={'k':k,'kappa':kappa},
                                loggerSettings={"logName":"logs/muller-brown"})
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    t0 = time.time()
    #Running
    tStepArr, alphaArr, stepsSinceReset, endsWithoutError = \
        nebObj.fire(dt,nIters,useLocal=False,earlyStop=False)
    t1 = time.time()
    print('Finished running NEB-LAP in %.3f s'%(t1-t0))
    
    pesAx.plot(*nebObj.allPts[-1].T,color='red',label='NEB-LAP')
    
    #Getting action values at each iteration
    actions = np.array([pyneb.TargetFunctions.action(p,pes)[0] for p in nebObj.allPts])
    
    #Getting action values interpolated at each iteration
    actionsInterpolated = np.zeros(nIters+2)
    for (pathIter,path) in enumerate(nebObj.allPts):
        interpPath = pyneb.InterpolatedPath(path)
        actionsInterpolated[pathIter] = \
            interpPath.compute_along_path(pyneb.TargetFunctions.action,500,
                                          [pes],{})[1][0]
    
    #Plotting action values
    fig, ax = plt.subplots()
    ax.plot(actions,label='Action')
    ax.plot(actionsInterpolated,label='Interpolated Action')
    ax.set(xlabel='Iteration',ylabel='Action',title='Action LAP')
    ax.legend()
    
    """NEB for the minimum energy path"""
    #NEB parameters
    nPts = 32
    nDims = 2
    k = 3
    kappa = 1
    
    #Optimization parameters
    dt = 0.01
    nIters = 500
    
    initialPath = np.array([np.linspace(-0.55,0.62,nPts),np.linspace(1.44,0.03,nPts)]).T
    
    #Setting up
    lap = pyneb.MinimumEnergyPath(pes,nPts,nDims,endpointSpringForce=False,
                                  endpointHarmonicForce=False,
                                  nebParams={'k':k,'kappa':kappa},
                                  loggerSettings={"logName":"logs/muller-brown"})
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    t0 = time.time()
    #Running
    tStepArr, alphaArr, stepsSinceReset, endsWithoutError = \
        nebObj.fire(dt,nIters,useLocal=False,earlyStop=False)
    t1 = time.time()
    print('Finished running NEB-MEP in %.3f s'%(t1-t0))
    
    pesAx.plot(*nebObj.allPts[-1].T,color='lime',label='NEB-MEP')
    
    #Getting action values at each iteration
    actions = np.array([pyneb.TargetFunctions.action(p,pes)[0] for p in nebObj.allPts])
    
    #Getting action values interpolated at each iteration
    actionsInterpolated = np.zeros(nIters+2)
    for (pathIter,path) in enumerate(nebObj.allPts):
        interpPath = pyneb.InterpolatedPath(path)
        actionsInterpolated[pathIter] = \
            interpPath.compute_along_path(pyneb.TargetFunctions.action,500,
                                          [pes],{})[1][0]
    
    #Plotting action values
    fig, ax = plt.subplots()
    ax.plot(actions,label='Action')
    ax.plot(actionsInterpolated,label='Interpolated Action')
    ax.set(xlabel='Iteration',ylabel='Action',title='Action MEP')
    ax.legend()
    
    #TODO: Finding and plotting critical points
    #maxima, minima, saddle = pyneb.get_crit_pnts(pes, nebObj.allPts[-1])
    
    pesAx.legend()