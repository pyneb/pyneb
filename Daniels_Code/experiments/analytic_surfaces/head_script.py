import sys, os
import pandas as pd

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

plt.style.use('science')

def camel_back(coords):
    origShape = coords.shape[:-1]
    
    x, y = coords.reshape((-1,2)).T
    res = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2)
        
    return res.reshape(origShape)

def asymm_camelback(coords):
    origShape = coords.shape[:-1]
    
    x, y = coords.reshape((-1,2)).T
    res = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2) + .5*y
        
    return res.reshape(origShape) 

def make_camelback():
    sp = np.array(pd.read_csv("sylvester_camelback.txt",header=None))
    
    dx = 0.1
    dy = 0.005
    x = np.arange(-2,2+dx,dx)
    y = np.arange(-1.25,1.25+dy,dy)
    
    cMeshTuple = np.meshgrid(x,y)
    cMeshArr = np.moveaxis(np.array(cMeshTuple),0,-1)
    
    zz = camel_back(cMeshArr)
    eGS = zz.min()
    zz = zz - eGS
    
    initPt = np.array([-1.700,0.790])
    finalPt = np.array([1.700,-0.790])
    
    if not os.path.isfile(os.getcwd()+"/logs/camelback.djk"):
        djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="camelback")
        pathInds, pathArr, _ = djk()
        interpPath = InterpolatedPath(pathArr)
        dist = interpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(camel_back,shift=eGS)])[1][0]
    else:
        djkLog = LoadDijkstraLog(os.getcwd()+"/logs/camelback.djk")
        pathInds = list(djkLog.allPathsIndsDict.values())[0]
        pathArr = list(djkLog.pathArrDict.values())[0]
        
        interpPath = InterpolatedPath(pathArr)
        dist = interpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(camel_back,shift=eGS)])[1][0]
    spInterpPath = InterpolatedPath(sp)
    spDist = spInterpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(camel_back,shift=eGS)])[1][0]
    
    print(dist)
    print(spDist)

    if not os.path.isfile(os.getcwd()+"/logs/camelback.dpm"):
        djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="camelback")
        dpm = DynamicProgramming(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="camelback")
        minIndsDict, minPathDict, distsDict = dpm()
        
        dpmPath = list(minPathDict.values())[0]
        dpmInterp = InterpolatedPath(dpmPath)
        dpmDist = dpmInterp.compute_along_path(TargetFunctions.action,500,\
                                               tfArgs=[shift_func(camel_back,shift=eGS)])[1][0]
    else:
        dpmLog = LoadDPMLogger(os.getcwd()+"/logs/camelback.dpm")
        # pathInds = list(dpmLog.pathIndsDict.values())[0]
        dpmPath = list(dpmLog.pathDict.values())[0]
        
        dpmInterp = InterpolatedPath(dpmPath)
        dpmDist = dpmInterp.compute_along_path(TargetFunctions.action,500,\
                                               tfArgs=[shift_func(camel_back,shift=eGS)])[1][0]    
    
    print(dpmDist)
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*cMeshTuple,zz.clip(0,5),cmap="Spectral_r",levels=100,extend="both")
    ax.contour(*cMeshTuple,zz.clip(0,5),levels=20,colors="gray")
    ax.plot(*pathArr.T,color="red",label="Dijstra: %.2f"%dist)
    ax.plot(*dpmPath.T,color="blue",label="DPM: %.2f"%dpmDist)
    ax.plot(*sp.T,linestyle="dashed",color="black",label="Sylvester: %.2f"%spDist)
    
    ax.legend(title="Action",frameon=True,fontsize=6,title_fontsize=6)
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
    cMeshArr = np.moveaxis(np.array(cMeshTuple),0,-1)
    
    zz = asymm_camelback(cMeshArr)
    eGS = zz.min()
    zz = zz - eGS
    
    initPt = np.array([-1.700,0.760])
    finalPt = np.array([1.700,-0.80])
    
    if not os.path.isfile(os.getcwd()+"/logs/asymm_camelback.djk"):
        djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="asymm_camelback")
        pathInds, pathArr, _ = djk()
        interpPath = InterpolatedPath(pathArr)
        dist = interpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(asymm_camelback,shift=eGS)])[1][0]
    else:
        djkLog = LoadDijkstraLog(os.getcwd()+"/logs/asymm_camelback.djk")
        pathInds = list(djkLog.allPathsIndsDict.values())[0]
        pathArr = list(djkLog.pathArrDict.values())[0]
        
        interpPath = InterpolatedPath(pathArr)
        dist = interpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(asymm_camelback,shift=eGS)])[1][0]
    spInterpPath = InterpolatedPath(sp)
    spDist = spInterpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(asymm_camelback,shift=eGS)])[1][0]
    
    print(dist)
    print(spDist)

    if not os.path.isfile(os.getcwd()+"/logs/asymm_camelback.dpm"):
        djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="asymm_camelback")
        dpm = DynamicProgramming(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="asymm_camelback")
        minIndsDict, minPathDict, distsDict = dpm()
        
        dpmPath = list(minPathDict.values())[0]
        dpmInterp = InterpolatedPath(dpmPath)
        dpmDist = dpmInterp.compute_along_path(TargetFunctions.action,500,\
                                               tfArgs=[shift_func(asymm_camelback,shift=eGS)])[1][0]
    else:
        dpmLog = LoadDPMLogger(os.getcwd()+"/logs/asymm_camelback.dpm")
        dpmPath = list(dpmLog.pathDict.values())[0]
        
        dpmInterp = InterpolatedPath(dpmPath)
        dpmDist = dpmInterp.compute_along_path(TargetFunctions.action,500,\
                                               tfArgs=[shift_func(asymm_camelback,shift=eGS)])[1][0]    
    
    print(dpmDist)
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*cMeshTuple,zz.clip(0,5),cmap="Spectral_r",levels=100,extend="both")
    ax.contour(*cMeshTuple,zz.clip(0,5),levels=20,colors="gray")
    ax.plot(*pathArr.T,color="red",label="Dijstra: %.2f"%dist)
    ax.plot(*dpmPath.T,color="blue",label="DPM: %.2f"%dpmDist)
    ax.plot(*sp.T,linestyle="dashed",color="black",label="Sylvester: %.2f"%spDist)
    
    ax.legend(title="Action",frameon=True,fontsize=6,title_fontsize=6)
    plt.colorbar(cf,ax=ax)
    
    fig.savefig("asymm_camelback.pdf",bbox_inches="tight")
    return None

# make_camelback()
make_asymm()