import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os

pynebDir = '..//../src/'
if pynebDir not in sys.path:
    sys.path.insert(0,pynebDir)
import pyneb

#TODO: For instructional purposes, can use this example to show different configurations
#of NEB (different target functions, gradients, and Verlet optimizers)

class LepsPot():
    def __init__(self,params={}):
        defaultParams = {"a":0.05,"b":0.8,"c":0.05,"dab":4.746,"dbc":4.746,\
                         "dac":3.445,"r0":0.742,"alpha":1.942,"rac":3.742,"kc":0.2025,\
                         "c_ho":1.154}
        for key in defaultParams.keys():
            if key in params:
                setattr(self,key,params[key])
            else:
                setattr(self,key,defaultParams[key])
    
    def _q(self,r,d,alpha,r0):
        return d/2*(3/2*np.exp(-2*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))
    
    def _j(self,r,d,alpha,r0):
        return d/4*(np.exp(-2*alpha*(r-r0)) - 6*np.exp(-alpha*(r-r0)))
    
    def leps_pot(self,rab,rbc):
        q = self._q
        j = self._j
                
        rac = rab + rbc
        
        vOut = q(rab,self.dab,self.alpha,self.r0)/(1+self.a) +\
            q(rbc,self.dbc,self.alpha,self.r0)/(1+self.b) +\
            q(rac,self.dac,self.alpha,self.r0)/(1+self.c)
        
        jab = j(rab,self.dab,self.alpha,self.r0)
        jbc = j(rbc,self.dbc,self.alpha,self.r0)
        jac = j(rac,self.dac,self.alpha,self.r0)
        
        jTerm = jab**2/(1+self.a)**2+jbc**2/(1+self.b)**2+jac**2/(1+self.c)**2
        jTerm = jTerm - jab*jbc/((1+self.a)*(1+self.b)) - jbc*jac/((1+self.b)*(1+self.c)) -\
            jab*jac/((1+self.a)*(1+self.c))
        
        vOut = vOut - np.sqrt(jTerm)
        
        return vOut
    
    def __call__(self,coords):
        """
        
        LEPs potential plus harmonic oscillator.
        Taken from Parameters are from Bruce J. Berne, Giovanni Ciccotti,David F. Coker, \
        Classical and Quantum Dynamics in Condensed Phase Simulations \
        Proceedings of the International School of Physics (1998) Chapter 16 \
        
        Call this function with a numpy array of rab and x:
        
            xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
            zz = leps_plus_ho(xx,yy),
        
        and plot it as
        
            fig, ax = plt.subplots()
            ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")
    
        """
        assert isinstance(coords,np.ndarray)
        assert coords.shape[-1] == 2
        
        ndims = coords.ndim
        rab, x = coords[(ndims-1)*(slice(None),)+(0,)], coords[(ndims-1)*(slice(None),)+(1,)]
        
        vOut = self.leps_pot(rab,self.rac-rab)
        vOut += 2*self.kc*(rab-(self.rac/2-x/self.c_ho))**2
        
        return vOut

if __name__ == '__main__':
    os.makedirs('logs',exist_ok=True)
    
    """Setting up LEPS potential on a grid"""
    leps = LepsPot()
    
    #Use unequal shapes to easily know shapes of arrays we're after
    x = np.linspace(.5,3.25,500)
    y = np.linspace(-3,3.05,490)
    
    xx, yy = np.meshgrid(x,y)
    zz = leps(np.swapaxes(np.array([xx,yy]),0,-1))
    
    gsInds = pyneb.SurfaceUtils.find_local_minimum(zz,searchPerc=[1,1])
    gsLoc = np.array((xx.T[gsInds],yy.T[gsInds]))
    gsEneg = leps(gsLoc)
    
    zz -= gsEneg
    
    pes = pyneb.shift_func(leps,gsEneg)
    
    """Plotting the LEPS function"""
    pesFig, pesAx = plt.subplots()
    cf = pesAx.contourf(xx,yy,zz.T,cmap='Spectral_r',levels=30)
    plt.colorbar(cf,ax=pesAx)
    pesAx.scatter(*gsLoc,marker='x',color='black')
    
    pesAx.set(xlabel='rAB',ylabel='x')
    
    """NEB parameters"""
    nPts = 42
    nDims = 2
    k = 2
    kappa = 1
    
    #Optimization parameters
    dt = 0.01
    nIters = 500
    
    initialPath = np.array([np.linspace(0.74,3,nPts),np.linspace(1.3,-1.3,nPts)]).T
    
    """NEB for the least action path"""
    #Setting up
    lap = pyneb.LeastActionPath(pes,nPts,nDims,endpointSpringForce=False,
                                endpointHarmonicForce=False,
                                target_func=pyneb.TargetFunctions.action_squared,
                                target_func_grad=pyneb.GradientApproximations().discrete_sqr_action_grad,
                                nebParams={'k':k,'kappa':kappa},
                                loggerSettings={"logName":"logs/leps"})
    nebObj = pyneb.VerletMinimization(lap,initialPath)
    
    t0 = time.time()
    #Running
    tStepArr, alphaArr, stepsSinceReset, endsWithoutError = \
        nebObj.fire2(dt,nIters,useLocal=False,earlyStop=False)
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
    #Setting up
    lap = pyneb.MinimumEnergyPath(pes,nPts,nDims,endpointSpringForce=False,
                                  endpointHarmonicForce=False,
                                  nebParams={'k':k,'kappa':kappa},
                                  loggerSettings={"logName":"logs/leps"})
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
    
    """Dijkstra for the LAP"""
    #Has to start and end on gridpoints near the real minima
    gridMinInds = pyneb.SurfaceUtils.find_all_local_minimum(zz)
    xMin, yMin = xx.T[gridMinInds],yy.T[gridMinInds]
    
    #We already know from looking at the surface which point is the beginning/end
    initLoc = np.array([xMin[0],yMin[0]])
    finalLoc = np.array([xMin[1],yMin[1]])
    djk = pyneb.Dijkstra(initLoc,(xx,yy),zz,
                         allowedEndpoints=finalLoc,
                         logLevel=1,fName='logs/leps')
    
    t0 = time.time()
    _, path, act = djk()
    t1 = time.time()
    print("Finished running Dijkstra's method in %.3f s"%(t1-t0))
    
    pesAx.plot(*path.T,color='pink',label='Dijkstra')
    
    pesAx.legend()
    