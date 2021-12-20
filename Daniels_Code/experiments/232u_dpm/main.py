import sys
import os
import time
import h5py
import pandas as pd

# from tabulate import tabulate
# from texttable import Texttable
# import latextable

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

plt.style.use('science')

def read_path(fname,returnHeads=False):
    df = pd.read_csv(fname,sep=",",index_col=None,header=None)
    
    firstRow = np.array(df.loc[0])
    try:
        firstRow = firstRow.astype(float)
        ret = np.array(df)
        heads = None
    except ValueError:
        ret = np.array(df.loc[1:]).astype(float)
        heads = df.loc[0]
    
    if returnHeads:
        ret = (ret,heads)
    
    return ret

def read_potential():
    fileDir = os.path.expanduser("~/Research/ActionMinimization/PES/")
    
    dsetsToGet = ["Q20","Q30","PES","B2020","B2030","B3030"]
    dsetsDict = {}
    
    h5File = h5py.File(fileDir+"232U.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    uniqueCoords = [np.unique(dsetsDict[key]) for key in coordNms]
    pesShape = [len(c) for c in uniqueCoords]
    dsetsDict = {key:dsetsDict[key].reshape(pesShape).T for key in dsetsDict.keys()}
    
    return dsetsDict

def run_dpm():
    dsetsDict = read_potential()
    
    qMesh = tuple([dsetsDict[key] for key in coordNms])
    gsInds = SurfaceUtils.find_local_minimum(dsetsDict["PES"])
    gsLoc = np.array([q[gsInds] for q in qMesh])
    
    dpm = DynamicProgramming(gsLoc,qMesh,dsetsDict["PES"],fName="default")
    minIndsDict, minPathDict, distsDict = dpm()
    
    minDist = np.inf
    for (endPt,dist) in distsDict.items():
        if dist < minDist:
            minPath = minPathDict[endPt]
            
    return None

def run_finer_dpm():
    dsetsDict = read_potential()
    
    uniqueCoords = [np.unique(dsetsDict[key]) for key in coordNms]
    pot_func = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])
    
    inertFuncsDict = {key:NDInterpWithBoundary(uniqueCoords,dsetsDict[key]) for\
                      key in ["B2020","B2030","B3030"]}
    
    q3 = np.arange(uniqueCoords[1][0],uniqueCoords[1][-1]+0.1,0.1)
    qMesh = np.meshgrid(uniqueCoords[0],q3)
    
    qMeshArr = np.moveaxis(np.array(qMesh),0,-1)
    zz = pot_func(qMeshArr)
    
    allowedEndpoints, _ = SurfaceUtils.find_endpoints_on_grid(qMesh,zz)
    
    gsInds = SurfaceUtils.find_local_minimum(zz)
    gsLoc = np.array([q[gsInds] for q in qMesh])
    
    zz[zz<0] = np.inf
    
    denseInertArr = {key:inertFuncsDict[key](qMeshArr) for key in inertFuncsDict.keys()}
    inertArr = np.array([[denseInertArr["B2020"],denseInertArr["B2030"]],\
                          [denseInertArr["B2030"],denseInertArr["B3030"]]])
    inertArr = np.moveaxis(inertArr,[0,1],[2,3])
    
    dpm = DynamicProgramming(gsLoc,qMesh,zz,inertArr=inertArr,fName="finer_with_mass_inf_cut",\
                             allowedEndpoints=allowedEndpoints)
                              
    dpm()
    
    return None

def plot_dpm():
    dsetsDict = read_potential()
    
    uniqueCoords = [np.unique(dsetsDict[key]) for key in coordNms]
    pot_func = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])
    massFuncsDict = {key:NDInterpWithBoundary(uniqueCoords,dsetsDict[key]) for\
                     key in ["B2020","B2030","B3030"]}
    mass_func = mass_funcs_to_array_func(massFuncsDict,["20","30"])
    
    # dpmInst = LoadDPMLogger("logs/finer.dpm")
    dpmInst = LoadDPMLogger("logs/finer_with_mass_inf_cut.dpm")
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*[dsetsDict[key] for key in coordNms],dsetsDict["PES"].clip(-5,30),\
                     cmap="Spectral_r",levels=np.arange(-5,31,1))
    plt.colorbar(cf,ax=ax)
    ax.scatter(*dpmInst.allowedEndpoints.T,color="black",s=0.25)
    
    sortInds = dpmInst.dists["endpoint"][:,0].argsort()
    distsArr = dpmInst.dists[sortInds]
    
    for i in range(distsArr.shape[0]):
        if i % 20 == 0:
            key = tuple(distsArr[i]["endpoint"])
            ax.plot(*dpmInst.pathDict[key].T)
    
    trueDists = np.zeros(distsArr.shape[0])
    for i in range(distsArr.shape[0]):
        p = InterpolatedPath(dpmInst.pathDict[tuple(distsArr[i]["endpoint"])])
        trueDists[i] = p.compute_along_path(TargetFunctions.action,500,\
                                            tfArgs=[pot_func],tfKWargs={"masses":mass_func})[1][0]
        
    distFig, distAx = plt.subplots()
    distAx.scatter(np.arange(distsArr.shape[0]),distsArr["dist"],s=4,label="Grid Distance")
    distAx.scatter(np.arange(distsArr.shape[0]),trueDists,s=4,label="Interpolated Distance")
    distAx.legend(frameon=True)
    
    minPathIter = np.argmin(trueDists)
    # minKey = tuple(distsArr[minPathIter]["endpoint"])
    # print(minKey)
    # ax.plot(*dpmInst.pathDict[minKey].T,"--")
    
    distAx.axvline(minPathIter,color="grey",ls="--")
    distAx.axhline(trueDists[minPathIter],color="grey",ls="--")
    distFig.savefig("distances.pdf",bbox_inches="tight")
    
    finalFig, finalAx = plt.subplots()
    cf = finalAx.contourf(*[dsetsDict[key] for key in coordNms],dsetsDict["PES"].clip(-5,30),\
                          cmap="Spectral_r",levels=np.arange(-5,31,1))
    plt.colorbar(cf,ax=finalAx)
    finalAx.scatter(*dpmInst.allowedEndpoints.T,color="black",s=0.25)
    endPt = tuple([298.,31.3])
    
    finalAx.plot(*dpmInst.pathDict[endPt].T,color="red",label="Daniel")
    # finalAx.plot(*dpmWMInst.pathDict[endPt].T,"--",color="red")
    # sp = read_path("Sylvester_path.txt")
    spm = read_path("232U_MAP_WMP.txt")
    # finalAx.plot(*sp.T,color="blue",label="Sylvester")
    finalAx.plot(*spm.T,"--",color="blue",label="Sylvester")
    
    for a in [ax,finalAx]:
        finalAx.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",title=r"${}^{232}$U")
    
    finalAx.legend(frameon=True,facecolor="white")
    
    finalFig.savefig("dpm_comparison_final.pdf",bbox_inches="tight")
    fig.savefig("borked_paths.pdf",bbox_inches="tight")
    
    return None

np.seterr(all="raise")

coordNms = ["Q20","Q30"]

plot_dpm()
# run_dpm()
#run_finer_dpm()

