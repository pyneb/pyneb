import sys, os
import pandas as pd

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

plt.style.use('science')

def camel_back(coords):
    origShape = coords.shape[1:]
    
    x, y = coords.reshape((2,-1))
    res = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2)
        
    return res.reshape(origShape)

def asymm_camelback(coords):
    origShape = coords.shape[1:]
    
    x, y = coords.reshape((2,-1))
    res = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2) + .5*y
        
    return res.reshape(origShape) 

def make_camelback():
    sp = np.array(pd.read_csv("sylvester_camelback.txt",header=None))
    
    dx = 0.1
    dy = 0.005
    x = np.arange(-2,2+dx,dx)
    y = np.arange(-1.25,1.25+dy,dy)
    
    cMeshTuple = np.meshgrid(x,y)
    zz = camel_back(np.array(cMeshTuple))
    zz = zz - zz.min()
    
    initPt = np.array([-1.700,0.790])
    finalPt = np.array([1.700,-0.790])
    
    djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="camelback")
    pathInds, pathArr, dist = djk()
    
    dpm = DynamicProgramming(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="camelback")
    minIndsDict, minPathDict, distsDict = dpm()
    dpmPath = list(minPathDict.values())[0]
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*cMeshTuple,zz.clip(0,5),cmap="Spectral_r",levels=100,extend="both")
    ax.contour(*cMeshTuple,zz.clip(0,5),levels=20,colors="gray")
    ax.plot(*pathArr.T,color="red")
    ax.plot(*dpmPath.T,color="blue")
    ax.plot(*sp.T,linestyle="dashed",color="black")
    
    plt.colorbar(cf,ax=ax)
    
    fig.savefig("camelback.pdf",bbox_inches="tight")
    return None

def make_asymm():
    sp = np.array(pd.read_csv("sylvester_asymm_camelback.txt",header=None))
    
    dx = 0.1
    dy = 0.005
    x = np.arange(-2,2+dx,dx)
    y = np.arange(-1.25,1.25+dy,dy)
    
    cMeshTuple = np.meshgrid(x,y)
    zz = asymm_camelback(np.array(cMeshTuple))
    zz = zz - zz.min()
    
    initPt = np.array([-1.700,0.760])
    finalPt = np.array([1.700,-0.80])
    
    djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="asymm_camelback")
    pathInds, pathArr, dist = djk()
    
    dpm = DynamicProgramming(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="asymm_camelback")
    minIndsDict, minPathDict, distsDict = dpm()
    dpmPath = list(minPathDict.values())[0]
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*cMeshTuple,zz.clip(0,5),cmap="Spectral_r",levels=100,extend="both")
    ax.contour(*cMeshTuple,zz.clip(0,5),levels=20,colors="gray")
    ax.plot(*pathArr.T,color="red")
    ax.plot(*dpmPath.T,color="blue")
    ax.plot(*sp.T,linestyle="dashed",color="black")
    
    plt.colorbar(cf,ax=ax)
    
    fig.savefig("asymm_camelback.pdf",bbox_inches="tight")
    return None

# make_camelback()
make_asymm()