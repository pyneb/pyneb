import sys
import os
import time
import h5py
import pandas as pd
import natsort

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

# def read_path(fname,returnHeads=False):
#     df = pd.read_csv(fname,sep=",",index_col=None,header=None)
    
#     firstRow = np.array(df.loc[0])
#     try:
#         firstRow = firstRow.astype(float)
#         ret = np.array(df)
#         heads = None
#     except ValueError:
#         ret = np.array(df.loc[1:]).astype(float)
#         heads = df.loc[0]
    
#     if returnHeads:
#         ret = (ret,heads)
    
#     return ret

def read_potential():
    fileDir = os.path.expanduser("~/Research/ActionMinimization/PES/")
    
    dsetsToGet = ["Q20","Q30","PES","B2020","B2030","B3030"]
    dsetsDict = {}
    
    h5File = h5py.File(fileDir+"232U.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    return dsetsDict

def main():
    coords = ["Q20","Q30"]

    dsetsDictIn = read_potential()
    uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
    pesShape = [len(c) for c in uniqueCoords]

    dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
    uniqueCoords = [np.unique(dsetsDict[c]) for c in coords]

    potential = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])

    pesRange = np.array([(c.min(),c.max()) for c in uniqueCoords])
    defaultStepSize = np.array([c[1]-c[0] for c in uniqueCoords])

    dq2 = 1
    dq3 = 0.1
    
    q2Unique = np.arange(pesRange[0,0],pesRange[0,1]+dq2,step=dq2)
    q3Unique = np.arange(pesRange[1,0],pesRange[1,1]+dq3,step=dq3)
    
    qMesh = np.meshgrid(q2Unique,q3Unique)
    qArr = np.moveaxis(np.array(qMesh),0,-1)
    
    zz = potential(qArr.reshape((-1,2))).reshape(qMesh[0].shape)
    
    gsInds = SurfaceUtils.find_local_minimum(zz)
    initPt = np.array([q[gsInds] for q in qMesh])
    
    allowedEndpoints = np.array([298,31.3])
    
    dpm = DynamicProgramming(initPt,qMesh,zz,allowedEndpoints=allowedEndpoints,\
                             fName="default")
    # print(zz.shape)
    indsDict, pathDict, distDict = dpm(500)
    print(distDict)
    
    fig, ax = plt.subplots()
    cf = ax.contourf(*qMesh,zz.clip(-5,30),cmap="Spectral_r")
    plt.colorbar(cf,ax=ax)
    
    ax.plot(*pathDict[tuple(allowedEndpoints)].T,marker=".",color="red")
    
    return None

# def analysis(toLoad):
#     coords = ["Q20","Q30"]

#     dsetsDictIn = read_potential()
#     uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
#     pesShape = [len(c) for c in uniqueCoords]

#     dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
#     dsetsDict = {key:dsetsDict[key][:26,50:101] for key in dsetsDict.keys()}
#     uniqueCoords = [np.unique(dsetsDict[c]) for c in coords]
    
#     fig, ax = plt.subplots()
#     cf = ax.contourf(dsetsDict["Q20"],dsetsDict["Q30"],dsetsDict["PES"].clip(5,15),\
#                      cmap="Spectral_r",levels=np.arange(5,15.5,0.5))
#     plt.colorbar(cf,ax=ax)
    
#     potential = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])
#     djkLog = LoadDijkstraLog("logs/"+toLoad+".djk")
    
#     colors = ["blue","brown","crimson","gold","green","grey","lime",\
#               "orchid","red","teal"]
    
#     sp = np.array(pd.read_csv("Sylvester_path.txt",header=None))
#     sp = sp[np.logical_and(sp[:,0]<=100,sp[:,0]>=50)]
#     sp = sp[sp[:,1]<=5]
#     ax.plot(*sp.T,color="black",ls=":")
#     plt.annotate("DP/LAP",xy=(95,3.3),xytext=(80,4),\
#                   bbox={"facecolor":"white","boxstyle":"round"},\
#                   arrowprops={"facecolor":"black","arrowstyle":"-|>"})
        
#     djkActions = np.zeros((10,2))
#     djkActionsAsIs = np.zeros((10,2))
#     arrIter = 0
#     for (keyIter, key) in enumerate(djkLog.pathArrDict.keys()):
#         if int(key[1]) == key[1] and key[0] == 100:#(key[1] % 2 == 0) and key[1]>0:
#             ax.plot(*djkLog.pathArrDict[key].T,color=colors[arrIter],\
#                     label="%i"%key[1])
#             interpPath = InterpolatedPath(djkLog.pathArrDict[key])
#             djkActions[arrIter,0] = key[1]
#             djkActions[arrIter,1] = interpPath.compute_along_path(TargetFunctions.action,200,\
#                                                                 tfArgs=[potential])[1][0]
#             djkActionsAsIs[arrIter,0] = key[1]
#             djkActionsAsIs[arrIter,1] = TargetFunctions.action(djkLog.pathArrDict[key],potential)[0]
            
#             arrIter += 1
    
#     lapActions = np.zeros((10,2))
#     lapActionsAsIs = np.zeros((10,2))
#     lapLogs = natsort.natsorted(["logs/velocity/"+f for f in \
#                                  os.listdir("logs/velocity") if f.endswith(".lap")])[1:]
#     for (lIter,l) in enumerate(lapLogs):
#         log = LoadForceLog(l)
#         ax.plot(*log.points[-1].T,ls="-.",color=colors[lIter])
#         interpPath = InterpolatedPath(log.points[-1])
#         lapActions[lIter,0] = log.points[-1,-1,1]
#         lapActions[lIter,1] = \
#             interpPath.compute_along_path(TargetFunctions.action,200,tfArgs=[potential])[1][0]
            
#         lapActionsAsIs[lIter,0] = log.points[-1,-1,1]
#         lapActionsAsIs[lIter,1] = TargetFunctions.action(log.points[-1],potential)[0]
        
#     suffix = toLoad
        
#     ax.legend(ncol=2,title=r"Final $Q_{30}$")
#     ax.set(xlabel=r"$Q_{20}$",ylabel=r"$Q_{30}$",title="232U Subset of PES",\
#            xlim=(50,100),ylim=(0,5))
#     fig.savefig("Path_comparison_"+suffix+".pdf",bbox_inches="tight")
    
#     # fig, ax = plt.subplots()
#     # ax.plot(*djkActions.T,marker=".",color="blue",label="Dijkstra")
#     # ax.plot(*lapActions.T,marker=".",color="orange",label="LAP")
    
#     # ax.plot(*djkActionsAsIs.T,marker=".",ls="-.",color="blue")
#     # ax.plot(*lapActionsAsIs.T,marker=".",ls="-.",color="orange")
    
#     # ax.grid()
#     # ax.set(xlabel=r"$Q_{30}$ Endpoint Value",ylabel="Action",title="Action Comparison")
#     # ax.legend()
    
#     # fig.savefig("Action_comparison_"+suffix+".pdf")
#     return None

# def plot_djk_dist():
#     coords = ["Q20","Q30"]

#     dsetsDictIn = read_potential()
#     uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
#     pesShape = [len(c) for c in uniqueCoords]

#     dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
#     dsetsDict = {key:dsetsDict[key][:26,50:101] for key in dsetsDict.keys()}
#     uniqueCoords = [np.unique(dsetsDict[c]) for c in coords]
    
#     potential = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])
#     djkLog = LoadDijkstraLog("logs/2021-10-26T13:17:09.473169.djk")
    
#     colors = ["blue","brown","crimson","gold","green","grey","lime",\
#               "orchid","red","teal"]
    
#     fig, ax = plt.subplots()
#     mesh = np.meshgrid(*djkLog.uniqueCoords)
#     a = ax.scatter(*mesh,c=djkLog.tentativeDistance.data.clip(30,80),s=512,marker="s")
#     # cf = ax.contourf(*djkLog.uniqueCoords,djkLog.tentativeDistance.data.clip(30,80),\
#     #                  cmap="Spectral_r",levels=20)
#     with np.printoptions(precision=3):
#         print(djkLog.tentativeDistance.shape)
#         print(djkLog.tentativeDistance.data[:6,9:19])
#         print(djkLog.tentativeDistance.data[:6,9:19].shape)
#     # plt.colorbar(cf,ax=ax)
        
#     sp = np.array(pd.read_csv("Sylvester_path.txt",header=None))
#     sp = sp[np.logical_and(sp[:,0]<=100,sp[:,0]>=50)]
#     sp = sp[sp[:,1]<=5]
#     ax.plot(*sp.T,color="black",ls=":")
#     plt.annotate("DP/LAP",xy=(95,3.3),xytext=(80,4),\
#                   bbox={"facecolor":"white","boxstyle":"round"},\
#                   arrowprops={"facecolor":"black","arrowstyle":"-|>"})
    
#     djkActions = np.zeros((10,2))
#     djkActionsAsIs = np.zeros((10,2))
#     arrIter = 0
#     for (keyIter, key) in enumerate(djkLog.pathArrDict.keys()):
#         if int(key[1]) == key[1] and key[0] == 100:#(key[1] % 2 == 0) and key[1]>0:
#             ax.plot(*djkLog.pathArrDict[key].T,color=colors[arrIter],\
#                     label="%i"%key[1])
#             interpPath = InterpolatedPath(djkLog.pathArrDict[key])
#             djkActions[arrIter,0] = key[1]
#             djkActions[arrIter,1] = interpPath.compute_along_path(TargetFunctions.action,200,\
#                                                                 tfArgs=[potential])[1][0]
#             djkActionsAsIs[arrIter,0] = key[1]
#             djkActionsAsIs[arrIter,1] = TargetFunctions.action(djkLog.pathArrDict[key],potential)[0]
            
#             arrIter += 1
    
#     lapActions = np.zeros((10,2))
#     lapActionsAsIs = np.zeros((10,2))
#     lapLogs = natsort.natsorted(["logs/velocity/"+f for f in \
#                                  os.listdir("logs/velocity") if f.endswith(".lap")])[1:]
#     for (lIter,l) in enumerate(lapLogs):
#         log = LoadForceLog(l)
#         ax.plot(*log.points[-1].T,ls="-.",color=colors[lIter])
#         interpPath = InterpolatedPath(log.points[-1])
#         lapActions[lIter,0] = log.points[-1,-1,1]
#         lapActions[lIter,1] = \
#             interpPath.compute_along_path(TargetFunctions.action,200,tfArgs=[potential])[1][0]
            
#         lapActionsAsIs[lIter,0] = log.points[-1,-1,1]
#         lapActionsAsIs[lIter,1] = TargetFunctions.action(log.points[-1],potential)[0]
            
#     ax.legend(ncol=2,title=r"Final $Q_{30}$",framealpha=1,edgecolor="black")
#     ax.set(xlabel=r"$Q_{20}$",ylabel=r"$Q_{30}$",title="232U Dijkstra Distance",\
#            xlim=(58,68),ylim=(0,0.5),xticks=np.arange(58,69,1))
#     ax.grid()
#     fig.savefig("Dijkstra_distance.pdf",bbox_inches="tight")
    
#     # fig, ax = plt.subplots()
#     # ax.plot(*djkActions.T,marker=".",color="blue",label="Dijkstra")
#     # ax.plot(*lapActions.T,marker=".",color="orange",label="LAP")
    
#     # ax.plot(*djkActionsAsIs.T,marker=".",ls="-.",color="blue")
#     # ax.plot(*lapActionsAsIs.T,marker=".",ls="-.",color="orange")
    
#     # ax.grid()
#     # ax.set(xlabel=r"$Q_{30}$ Endpoint Value",ylabel="Action",title="Action Comparison")
#     # ax.legend()
    
#     # fig.savefig("Action_comparison_"+suffix+".pdf")
#     return None

# def lap_comparison():
#     action = TargetFunctions.action

#     coords = ["Q20","Q30"]

#     dsetsDictIn = read_potential()
#     uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
#     pesShape = [len(c) for c in uniqueCoords]

#     dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
#     dsetsDict = {key:dsetsDict[key][:101,:201] for key in dsetsDict.keys()}
#     uniqueCoords = [np.unique(dsetsDict[c]) for c in coords]

#     potential = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])
#     initialPoint = (25.,0.)
#     nPts = 30
    
#     actFig, actAx = plt.subplots()
#     pesFig, pesAx = plt.subplots()
#     pesAx.contourf(*[dsetsDict[q] for q in coords],dsetsDict["PES"])
    
#     endPts = np.array([[200,q3] for q3 in np.arange(0,21,2)],dtype=float)
#     for i in range(endPts.shape[0]):
#         logNm = "Endpoint_"+str(i)+".lap"
        
#         lapInst = LeastActionPath(potential,nPts,2,endpointSpringForce=False,\
#                                   endpointHarmonicForce=False,\
#                                   loggerSettings={"logName":logNm,"writeFreq":50})
        
#         initialPoints = np.array([np.linspace(initialPoint[j],endPts[i,j],nPts)\
#                                   for j in range(2)]).T
#         verletInstance = VerletMinimization(lapInst,initialPoints)
#         # allPts, _, _ = verletInstance.velocity_verlet(0.1,1000)
#         verletInstance.fire(0.01,1000,fireParams={"dtMin":0.001,"dtMax":0.05},useLocal=True)
        
#         # actions = np.array([action(pts,potential)[0] for pts in allPts])
#         actions = np.array([action(pts,potential)[0] for pts in verletInstance.allPts])
#         actAx.plot(actions/(actions[-1]+10**(-2)))
        
#         pesAx.plot(*verletInstance.allPts[-1].T)
#         # pesAx.plot(*allPts[-1].T)
    
#     actAx.set(xlabel="Iterations",ylabel="Action (Over Final Action)")
#     actFig.savefig("LAP_Comparison.pdf")
    
#     return None

# def plot_inset_pes():
#     coords = ["Q20","Q30"]

#     dsetsDictIn = read_potential()
#     uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
#     pesShape = [len(c) for c in uniqueCoords]

#     dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
#     dsetsDict = {key:dsetsDict[key][:26,50:101] for key in dsetsDict.keys()}
#     uniqueCoords = [np.unique(dsetsDict[c]) for c in coords]
    
#     fig, ax = plt.subplots()
#     cf = ax.contourf(dsetsDict["Q20"],dsetsDict["Q30"],dsetsDict["PES"].clip(5,15),\
#                      cmap="Spectral_r",levels=np.arange(5,15.5,0.5))
#     plt.colorbar(cf,ax=ax)
    
#     return None

# def small_dijkstra():
#     action = TargetFunctions.action

#     coords = ["Q20","Q30"]

#     dsetsDictIn = read_potential()
#     uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
#     pesShape = [len(c) for c in uniqueCoords]

#     dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}
#     # dsetsDict = {key:dsetsDict[key][:26,50:101] for key in dsetsDict.keys()}
#     uniqueCoords = [np.unique(dsetsDict[c]) for c in coords]

#     potential = NDInterpWithBoundary(uniqueCoords,dsetsDict["PES"])

#     pesRange = np.array([(c.min(),c.max()) for c in uniqueCoords])
#     defaultStepSize = np.array([c[1]-c[0] for c in uniqueCoords])

#     dq2 = 0.5
#     dq3 = 0.1
    
#     q2Unique = np.arange(215.,220+dq2,step=dq2)
#     q3Unique = np.arange(25.,26+dq3,step=dq3)
    
#     qMesh = np.meshgrid(q2Unique,q3Unique)
#     qArr = np.moveaxis(np.array(qMesh),0,-1)
    
#     zz = potential(qArr.reshape((-1,2))).reshape(qMesh[0].shape)
#     # print(zz[0,25])
#     # gsInds = SurfaceUtils.find_local_minimum(zz)
#     # initPt = np.array([q[gsInds] for q in qMesh])
#     # print(initPt)
#     initPt = np.array([216.,25.1])
    
#     # allowedEndpoints = np.array([[50.,q] for q in q3Unique if q.is_integer()])
#     allowedEndpoints = np.array([220,25.9])
    
#     djk = Dijkstra(initPt,qMesh,zz,allowedEndpoints=allowedEndpoints,fName="Mini",\
#                    logLevel=2)
    
#     fig, ax = plt.subplots()
#     cf = ax.contourf(*qMesh,zz.clip(-5,30),cmap="Spectral_r")
#     plt.colorbar(cf,ax=ax)
    
#     t0 = time.time()
#     _, pathsDict, distDict = djk(returnAll=True)
#     t1 = time.time()
#     runTime = t1 - t0
    
#     sp = np.array(pd.read_csv("Sylvester_path.txt",header=None))
#     sp = sp[np.logical_and(sp[:,1]>=25,sp[:,1]<=26)]
    
#     ax.scatter(*initPt,marker="x",color="red")
#     for (key,val) in pathsDict.items():
#         ax.plot(val[:,0],val[:,1],".-",label="Dijkstra: %.3f"%distDict[key])
#     ax.plot(*sp.T,".-",label="DP: %.3f"%action(sp,potential)[0])
#     ax.legend()
    
#     ax.set(xlabel="Q20",ylabel="Q30",label="232U",xticks=q2Unique,yticks=q3Unique)
#     ax.grid()
    
#     testPath = np.array([[i,0] for i in q2Unique])
#     print("Test action: %.3f"%action(testPath,potential)[0])
#     print("Actions dict:\n",distDict)
    
#     djkLog = LoadDijkstraLog("logs/Mini.djk")
#     # with np.printoptions(precision=3):
#     #     print(djkLog.tentativeDistance.shape)
#     #     print(djkLog.tentativeDistance.data[:5,8:15])
#     ax.set(xlim=(215,220),ylim=(25,26.1))
    
#     fig.savefig("Mini_djk_2.pdf",bbox_inches="tight")
#     return None

np.seterr(all="raise")
main()