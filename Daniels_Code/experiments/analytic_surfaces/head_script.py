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
    res = (4. - 2.1 * x**2 + 1/3 * x**4) * x**2 + x * y + 4*(y**2 - 1) * y**2
        
    return res.reshape(origShape)

def asymm_camelback(coords):
    origShape = coords.shape[:-1]
    
    x, y = coords.reshape((-1,2)).T
    res = (4 - 2.1*(x**2) + (1/3) * (x**4))*x**2 + x*y + 4*((y**2) - 1)*(y**2) + .5*y
        
    return res.reshape(origShape)

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

def make_camelback():
    sp = np.array(pd.read_csv("sylvester_camelback.txt",header=None))
    
    dx = 0.1
    dy = 0.005
    x = np.arange(-2,2+dx,dx)
    y = np.arange(-1.25,1.25+dy,dy)
    
    cMeshTuple = np.meshgrid(x,y)
    cMeshArr = np.moveaxis(np.array(cMeshTuple),0,-1)
    print(cMeshArr.shape)
    
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
    ax.plot(*pathArr.T,color="red",label="Dijkstra: %.2f"%dist)
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
    ax.plot(*pathArr.T,color="red",label="Dijkstra: %.2f"%dist)
    ax.plot(*dpmPath.T,color="blue",label="DPM: %.2f"%dpmDist)
    ax.plot(*sp.T,linestyle="dashed",color="black",label="Sylvester: %.2f"%spDist)
    
    ax.legend(title="Action",frameon=True,fontsize=6,title_fontsize=6)
    plt.colorbar(cf,ax=ax)
    
    fig.savefig("asymm_camelback.pdf",bbox_inches="tight")
    return None

def make_muller_brown():
    sp = np.array(pd.read_csv("sylvester_mullerbrown.txt",header=None))
    
    dx = 0.01
    x = np.arange(-1.5,1+dx,dx)
    dy = 0.01
    y = np.arange(-0.25,1.75+dy,dy)
    
    cMeshTuple = np.meshgrid(x,y)
    cMeshArr = np.moveaxis(np.array(cMeshTuple),0,-1)
    
    zz = muller_brown(cMeshArr)
    eGS = zz.min()
    zz = zz - eGS
    
    fig, ax = plt.subplots()
    levels = np.arange(10,190,20)
    cf = ax.contourf(*cMeshTuple,zz.clip(0,185),cmap="Spectral_r",levels=50,extend="both")
    ax.contour(*cMeshTuple,zz.clip(0,185),levels=levels,colors="gray")
    plt.colorbar(cf,ax=ax)
    
    initPt = np.array([-0.580,1.410])
    finalPt = np.array([0.620,0.020])
    
    if not os.path.isfile(os.getcwd()+"/logs/muller_brown.dpm"):
        dpm = DynamicProgramming(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="muller_brown")
        minIndsDict, minPathDict, distsDict = dpm()
        
        dpmPath = list(minPathDict.values())[0]
        dpmInterp = InterpolatedPath(dpmPath)
        dpmDist = dpmInterp.compute_along_path(TargetFunctions.action,500,\
                                               tfArgs=[shift_func(muller_brown,shift=eGS)])[1][0]
    else:
        dpmLog = LoadDPMLogger(os.getcwd()+"/logs/muller_brown.dpm")
        dpmPath = list(dpmLog.pathDict.values())[0]
        
        dpmInterp = InterpolatedPath(dpmPath)
        dpmDist = dpmInterp.compute_along_path(TargetFunctions.action,500,\
                                               tfArgs=[shift_func(muller_brown,shift=eGS)])[1][0]
            
    if not os.path.isfile(os.getcwd()+"/logs/muller_brown.djk"):
        djk = Dijkstra(initPt,cMeshTuple,zz,allowedEndpoints=finalPt,fName="muller_brown")
        pathInds, pathArr, _ = djk()
        interpPath = InterpolatedPath(pathArr)
        dist = interpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(muller_brown,shift=eGS)])[1][0]
    else:
        djkLog = LoadDijkstraLog(os.getcwd()+"/logs/muller_brown.djk")
        pathInds = list(djkLog.allPathsIndsDict.values())[0]
        pathArr = list(djkLog.pathArrDict.values())[0]
        
        interpPath = InterpolatedPath(pathArr)
        dist = interpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(muller_brown,shift=eGS)])[1][0]
            
    print("Dijkstra: ",dist)
    print("DPM: ",dpmDist)
    spInterpPath = InterpolatedPath(sp)
    spDist = spInterpPath.compute_along_path(TargetFunctions.action,500,\
                                             tfArgs=[shift_func(muller_brown,shift=eGS)])[1][0]
    print("SP DPM: ",spDist)
    
    ax.plot(*pathArr.T,color="red",label="Dijkstra: %.2f"%dist)
    ax.plot(*dpmPath.T,color="blue",label="DPM: %.2f"%dpmDist)
    ax.plot(*sp.T,linestyle="dashed",color="black",label="Sylvester: %.2f"%spDist)
    
    ax.set(xticks=np.arange(-1.5,1.5,0.5),yticks=np.arange(-0.25,2,0.25))
    ax.legend(title="Action",frameon=True,fontsize=6,title_fontsize=6)
    fig.savefig("muller_brown.pdf",bbox_inches="tight")
    return None

# make_camelback()
# make_asymm()
make_muller_brown()