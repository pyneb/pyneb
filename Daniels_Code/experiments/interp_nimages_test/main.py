import sys, os
import pandas as pd

pyNebDir = os.path.expanduser(os.getcwd()+"//..//..//../py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

plt.style.use('science')

from tabulate import tabulate
from texttable import Texttable
import latextable

def camel_back(coords):
    origShape = coords.shape[:-1]
    
    x, y = coords.reshape((-1,2)).T
    res = (4. - 2.1 * x**2 + 1/3 * x**4) * x**2 + x * y + 4*(y**2 - 1) * y**2
        
    return res.reshape(origShape)

def asymm_camelback(coords):
    res = camel_back(coords) + 0.5*coords[(coords.ndim-1)*(slice(None),)+(1,)]
        
    return res

def muller_brown(coords):
    A = np.array([-200.,-100,-170,15])
    a = np.array([-1.,-1,-6.5,0.7])
    b = np.array([0.,0,11,0.6])
    c = np.array([-10.,-10,-6.5,0.7])
    x0 = np.array([1.,0,-0.5,-1])
    y0 = np.array([0.,0.5,1.5,1])
    
    slc = (coords.ndim-1)*(slice(None),)
    x = coords[slc+(0,)]
    y = coords[slc+(1,)]
    
    res = np.zeros(x.shape)
    for i in range(4):
        expTerm = a[i]*(x-x0[i])**2+b[i]*(x-x0[i])*(y-y0[i])+c[i]*(y-y0[i])**2
        res += A[i]*np.exp(expTerm)
    
    return res

initPt = np.array([-0.550,1.440])
finalPt = np.array([0.600,0.030])
    
dxVals = np.logspace(-3,-1,20)
nImages = np.arange(20,220,20)

for dx in dxVals:
    for n in nImages:
        dy = dx
        
        x = np.arange(-1.5,1+dx,dx)
        y = np.arange(-0.25,1.75+dy,dy)
        
        cMeshTuple = np.meshgrid(x,y)
        cMeshArr = np.moveaxis(np.array(cMeshTuple),0,-1)
        zz = muller_brown(cMeshArr)
        eGS = zz.min()
        zz = zz - eGS
        
        pes_func = NDInterpWithBoundary((x,y),zz)
        
        initialPoints = np.array([np.linspace(initPt[i],finalPt[i],n) for\
                                  i in range(2)]).T
        
        lap = LeastActionPath(pes_func,n,2,endpointSpringForce=False,\
                              endpointHarmonicForce=False)
        neb = VerletMinimization(lap,initialPoints)
        
        neb.fire(0.2,2000,earlyStop=True,fireParams={"maxmove":0.1*np.ones(2)},
                 earlyStopParams={"stabPerc":10**(-3)})
        
        acts = np.array([TargetFunctions.action(p,pes_func)[0] for p in neb.allPts])
        
        h5File = h5py.File(lap.logger.fileName,"a")
        h5File.create_dataset("target_func_values",data=acts)
        h5File.close()
        