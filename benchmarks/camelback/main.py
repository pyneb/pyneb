from context import pyneb
 
import numpy as np
import matplotlib.pyplot as plt
    
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

if __name__ == "__main__":
    #Setting up camelback potential on a grid
    x = np.arange(-2,2,0.05)
    y = np.arange(-1.25,1.25,0.05)
    coords = [x,y]
    
    xx, yy = np.meshgrid(x,y)
    zz = camelback(np.swapaxes(np.array([xx,yy]),0,-1))
    
    pesFig, pesAx = plt.subplots()
    cf = pesAx.contourf(xx,yy,zz.T,cmap="Spectral_r",extend="both",levels=45)
    plt.colorbar(cf,ax=pesAx)
    
    pesAx.set(xlabel="x",ylabel="y",title="Camelback Potential")
    
    actFig, actAx = plt.subplots()
    actAx.set(xlabel="Iteration",ylabel="Action",title="Action vs Iteration Number")
    
    nPts = 52
    nDims = 2
    
    tStep = 0.1
    nIters = 100
    
    #Simple Verlet minimization
    loggerSettings = {"logName":"verlet"}
    lap = pyneb.LeastActionPath(camelback,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,loggerSettings=loggerSettings)
    
    initialPath = np.array([np.linspace(-1.7,1.7,nPts),np.linspace(0.79,-0.79,nPts)]).T
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    nebObj.velocity_verlet(tStep,nIters)
    
    acts = [pyneb.TargetFunctions.action(p,camelback)[0] for p in nebObj.allPts]
    
    pesAx.plot(*nebObj.allPts[-1].T,marker=".",label="Velocity Verlet")
    actAx.plot(acts,label="Velocity Verlet")
    
    #Local FIRE
    loggerSettings = {"logName":"local_fire"}
    lap = pyneb.LeastActionPath(camelback,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,loggerSettings=loggerSettings)
    
    initialPath = np.array([np.linspace(-1.7,1.7,nPts),np.linspace(0.79,-0.79,nPts)]).T
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    nebObj.fire(tStep,nIters,useLocal=True,earlyStop=False)
    
    acts = [pyneb.TargetFunctions.action(p,camelback)[0] for p in nebObj.allPts]
    
    pesAx.plot(*nebObj.allPts[-1].T,marker=".",label="Local FIRE")
    actAx.plot(acts,label="Local FIRE")
    
    #Global FIRE
    loggerSettings = {"logName":"global_fire"}
    lap = pyneb.LeastActionPath(camelback,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,loggerSettings=loggerSettings)#,
                                # target_func=pyneb.TargetFunctions.action_squared)#,
                                # nebParams={"k":1})
    
    initialPath = np.array([np.linspace(-1.7,1.7,nPts),np.linspace(0.79,-0.79,nPts)]).T
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    nebObj.fire(tStep,nIters,useLocal=False,earlyStop=False)#,
                # fireParams={"maxmove":np.array([0.01,0.01])})
    
    acts = [pyneb.TargetFunctions.action(p,camelback)[0] for p in nebObj.allPts]
    
    pesAx.plot(*nebObj.allPts[-1].T,marker=".",label="Global FIRE",color="red")
    actAx.plot(acts,label="Global FIRE")
    
    #Global FIRE 2
    loggerSettings = {"logName":"global_fire2"}
    lap = pyneb.LeastActionPath(camelback,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,loggerSettings=loggerSettings)
    
    initialPath = np.array([np.linspace(-1.7,1.7,nPts),np.linspace(0.79,-0.79,nPts)]).T
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    tStepArr, alphaArr, stepsSinceReset = nebObj.fire2(tStep,nIters,useLocal=False,earlyStop=False,
                                                       fireParams={"maxmove":np.array([0.1,0.1])})
    
    acts = [pyneb.TargetFunctions.action(p,camelback)[0] for p in nebObj.allPts]
    
    pesAx.plot(*nebObj.allPts[-1].T,marker=".",label="Global FIRE 2")
    actAx.plot(acts,label="Global FIRE 2")
    
    pesAx.legend()
    actAx.legend()
    
    # fig, ax = plt.subplots()
    # ax.plot(tStepArr)
    # ax.set(title="dt")
    
    # vFig, vAx = plt.subplots()
    # for f in nebObj.allVelocities:
    #     vAx.plot(*(f.T - nebObj.allVelocities[0].T))
    # vAx.set(title="Velocity")
    
    # forceFig, forceAx = plt.subplots()
    # for f in nebObj.allForces:
    #     forceAx.plot(*(f.T - nebObj.allForces[0].T))
    # forceAx.set(title="Force")
