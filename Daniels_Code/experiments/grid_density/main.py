import sys
import os
import time
import h5py
import pandas as pd

from tabulate import tabulate
from texttable import Texttable
import latextable

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb.py_neb import *

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
    
    return dsetsDict

def loop():
    h5File = h5py.File("output.h5","w")
    dsetIter = 0
    for q2 in q20StepSizes:
        for q3 in q30StepSizes:
            q2Unique = np.arange(pesRange[0,0],pesRange[0,1]+q2,step=q2)
            q3Unique = np.arange(pesRange[1,0],pesRange[1,1]+q3,step=q3)
            
            qMesh = np.meshgrid(q2Unique,q3Unique)
            qArr = np.moveaxis(np.array(qMesh),0,-1)
            
            zz = potential(qArr.reshape((-1,2))).reshape(qMesh[0].shape)
            
            gsInds = SurfaceUtils.find_local_minimum(zz)
            gsLoc = np.array([q[gsInds] for q in qMesh])
            
            djk = Dijkstra(gsLoc,qMesh,zz)
            t0 = time.time()
            _, pathArrDict, distDict = djk()
            t1 = time.time()
            runTime = t1 - t0
            
            minEndpt = djk.minimum_endpoint(distDict)
            
            fig, ax = plt.subplots()
            cf = ax.contourf(*qMesh,zz.clip(-5,30),cmap="Spectral_r")
            plt.colorbar(cf,ax=ax)
            
            ax.contour(*qMesh,zz,levels=[0],colors=["black"])
            ax.scatter(*gsLoc,marker="x",color="red")
            ax.scatter(*minEndpt,marker="^",color="red")
            
            ax.plot(pathArrDict[minEndpt][:,0],pathArrDict[minEndpt][:,1],color="blue")
            
            h5File.create_group(str(dsetIter))
            h5File[str(dsetIter)].create_dataset("q2Vals",data=q2Unique)
            h5File[str(dsetIter)].create_dataset("q3Vals",data=q3Unique)
            h5File[str(dsetIter)].create_dataset("zz",data=zz)
            h5File[str(dsetIter)].create_dataset("path",data=pathArrDict[minEndpt])
            h5File[str(dsetIter)].attrs.create("q_spacing",(q2,q3))
            h5File[str(dsetIter)].attrs.create("run_time",runTime)
            
            dsetIter += 1
            
    h5File.close()
    return None

def plot():
    colors = {1.:"red",2.:"blue",5.:"green",10.:"orange"}
    styles = {0.05:"-",0.1:"--",0.2:"-.",0.5:":"}
    
    pathsDict = {}
    spacing = {}
    
    h5File = h5py.File("output.h5","r")
    for i in range(16):
        pathsDict[i] = np.array(h5File[str(i)+"/path"])
        spacing[i] = np.array(h5File[str(i)].attrs["q_spacing"])
    
    h5File.close()
    sp = read_path("Sylvester_path.txt")
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*[dsetsDict[c] for c in coords],dsetsDict["PES"].clip(-5,30),\
                     cmap="Spectral_r")
    plt.colorbar(cf,ax=ax)
    
    for i in range(16):
        if spacing[i][0] != 10.:
            ax.plot(pathsDict[i][:,0],pathsDict[i][:,1],\
                    color=colors[spacing[i][0]],ls=styles[spacing[i][1]])
        
    ax.plot(sp[:,0],sp[:,1],color="pink")
    ax.set(xlabel="Q20",ylabel="Q30",title="232U Dijkstra Paths",xlim=(0,325),\
           ylim=(0,40))
    
    q2Artist = [plt.Line2D((0,1),(0,0),color=colors[i]) for i in [1.,2.,5.]]
    l1 = plt.legend(q2Artist,[str(i) for i in [1.,2.,5.]],title=r"$\Delta Q_{20}$ (b)")
    
    q3Artist = [plt.Line2D((0,1),(0,0),color="k",linestyle=l) for l in styles.values()]
    l2 = plt.legend(q3Artist,[str(i) for i in styles.keys()],\
                    title=r"$\Delta Q_{30}$ (b${}^{3/2}$)",loc="lower right")
    ax.add_artist(l1)
    ax.add_artist(l2)
    
    fig.savefig("Path_Differences.pdf",bbox_inches="tight")
    
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*[dsetsDict[c] for c in coords],dsetsDict["PES"].clip(-5,30),\
                     cmap="Spectral_r")
    plt.colorbar(cf,ax=ax)
    
    sylvesterInd = 1.
    myAct, _, _ = action(pathsDict[sylvesterInd],potential)
    
    myInterpPath = InterpolatedPath(pathsDict[sylvesterInd])
    myDensePath, (myDenseAct,_,_) = myInterpPath.compute_along_path(action,500,tfArgs=[potential])
    ax.plot(pathsDict[sylvesterInd][:,0],pathsDict[sylvesterInd][:,1],color="red",\
            label="Daniel: %.3f" % myAct)
    ax.plot(myDensePath[:,0],myDensePath[:,1],color="red",ls="--",\
            label="Daniel Interp: %.3f" % myDenseAct)
    
    spAct, _, _ = action(sp,potential)
    spInterpPath = InterpolatedPath(sp)
    spDensePath, (spDenseAct,_,_) = spInterpPath.compute_along_path(action,500,tfArgs=[potential])
    ax.plot(sp[:,0],sp[:,1],color="blue",label="Sylvester: %.3f" % spAct)
    ax.plot(spDensePath[:,0],spDensePath[:,1],color="blue",ls="--",\
            label="Sylvester Interp: %.3f" % spDenseAct)
        
    ax.legend()
    fig.savefig("Same_spacing_paths.pdf",bbox_inches="tight")
    
    """ Making tables """
    
    rows = [["Spacing"] + [str(i) for i in colors.keys()]]
    for q3 in styles.keys():
        nextRow = [str(q3)]
        for q2 in colors.keys():
            for (key, arr) in spacing.items():
                if np.array_equal(arr,np.array([q2,q3])):
                    path = pathsDict[key]
            iPath = InterpolatedPath(path)
            _, (act,_,_) = iPath.compute_along_path(action,500,tfArgs=[potential])
            nextRow.append("{:5.1f}".format(act))
        rows.append(nextRow)
    
    table = Texttable()
    table.set_cols_align(5*["c"])
    # table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)
    print(latextable.draw_latex(table))
    
    
    
    return None

action = TargetFunctions.action

coords = ["Q20","Q30"]

dsetsDictIn = read_potential()
uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
pesShape = [len(c) for c in uniqueCoords]

dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}

# potential = NDInterpWithBoundary_experimental(uniqueCoords,dsetsDict["PES"])
potential = RectBivariateSplineWrapper(*uniqueCoords,dsetsDict["PES"].T).function

pesRange = np.array([(c.min(),c.max()) for c in uniqueCoords])
defaultStepSize = np.array([c[1]-c[0] for c in uniqueCoords])

q20StepSizes = np.array([1,2,5,10]) #Want to do with 2 and 5 as well
q30StepSizes = np.array([0.05,0.1,0.2,0.5]) #Would like to also do 0.01

loop()
plot()