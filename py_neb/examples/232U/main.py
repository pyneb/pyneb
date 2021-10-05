import sys
import os
import time
import h5py

pyNebDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..//.."))
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

def read_potential():
    dsetsToGet = ["Q20","Q30","PES","B2020","B2030","B3030"]
    dsetsDict = {}
    
    h5File = h5py.File("232U.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    return dsetsDict

#%%Setting up the potential on a grid
coords = ["Q20","Q30"]

dsetsDictIn = read_potential()
uniqueCoords = [np.unique(dsetsDictIn[c]) for c in coords]
pesShape = [len(c) for c in uniqueCoords]

dsetsDict = {key:dsetsDictIn[key].reshape(pesShape).T for key in dsetsDictIn.keys()}

coordMeshTuple = tuple([dsetsDict[c] for c in coords])
zz = dsetsDict["PES"]

#Shifting the ground state to be positive. Note that the PES is not always positive
gsInds = SurfaceUtils.find_local_minimum(zz)
gsLoc = np.array([c[gsInds] for c in coordMeshTuple])
zz -= zz[gsInds]

potential = RectBivariateSplineWrapper(*uniqueCoords,zz.T).func_wrapper()

#%%Finding endpoints
allowedEndpoints, allowedIndices = \
    find_endpoints_on_grid(coordMeshTuple,zz)

start = gsLoc
end = np.array([298.,31.2]) #Selected with prior knowledge of the PES to be near
                            #the outer turning line

#%%Getting least action path (LAP) with Dijkstra's algorithm
dijkstra = Dijkstra(start,coordMeshTuple,zz,allowedEndpoints=allowedEndpoints)
t0 = time.time()
_, pathArrDict, actionDict = dijkstra()
t1 = time.time()

dijkstraTime = t1 - t0

endptTuple = tuple(end.tolist())
path = pathArrDict[endptTuple]
dijkstraAction = actionDict[endptTuple]

#%%Getting LAP with NEB
nPts = 30
nDims = 2

lap = LeastActionPath(potential,nPts,nDims,endpointSpringForce=False,\
                      endpointHarmonicForce=False)

initialPath = \
    np.vstack([np.linspace(start[cIter],end[cIter],nPts) for cIter in range(nDims)]).T
tStep = 0.5
maxIters = 1000
useLocal = True

t0 = time.time()
verletLAP = VerletMinimization(lap,initialPath)
verletLAP.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
verletLAPAction = np.array([action(path,potential)[0] for path in verletLAP.allPts])
t1 = time.time()
lapTime = t1 - t0

#%%Getting minimum energy path with NEB
nPts = 30
nDims = 2

lap = MinimumEnergyPath(potential,nPts,nDims,endpointSpringForce=False,\
                        endpointHarmonicForce=False)

initialPath = \
    np.vstack([np.linspace(start[cIter],end[cIter],nPts) for cIter in range(nDims)]).T
tStep = 0.5
maxIters = 1000
useLocal = True

t0 = time.time()
verletMEP = VerletMinimization(lap,initialPath)
verletMEP.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
verletMEPAction = np.array([action(path,potential)[0] for path in verletMEP.allPts])
t1 = time.time()
mepTime = t1 - t0

#%%Printing action values and run times
print("Action along path: %.3f" % dijkstraAction)
print("Action along Verlet LAP: %.3f" % verletLAPAction[-1])
print("Action along Verlet MEP: %.3f" % verletMEPAction[-1])

print("Dijkstra run time: %.3f s" % dijkstraTime)
print("LAP run time: %.3f s" % lapTime)
print("MEP run time: %.3f s" % mepTime)

#%%Writing paths to text files
np.savetxt("Dijkstra_path.txt",path,delimiter=",")
np.savetxt("LAP.txt",verletLAP.allPts[-1],delimiter=",")
np.savetxt("MEP.txt",verletMEP.allPts[-1],delimiter=",")

#%%Plotting results
fig, ax = plt.subplots()
cf = ax.contourf(*coordMeshTuple,zz.clip(-5,50),levels=np.arange(-5,51,1))
ax.contour(*coordMeshTuple,zz,levels=[0],colors=["white"])
plt.colorbar(cf,ax=ax)

ax.scatter(*start,color="red",marker="x")
ax.scatter(*end,color="red",marker="^")
# 
ax.set(xlabel="Q20",ylabel="Q30",title="232U")

ax.plot(path[:,0],path[:,1],color="orange",label="Dijkstra (%.3f)" % dijkstraAction)
ax.plot(verletLAP.allPts[-1,:,0],verletLAP.allPts[-1,:,1],marker=".",color="lime",\
        label="LAP (%.3f)" % verletLAPAction[-1])
ax.plot(verletMEP.allPts[-1,:,0],verletMEP.allPts[-1,:,1],marker=".",color="cyan",\
        label="MEP (%.3f)" % verletMEPAction[-1])

ax.legend()

fig.savefig("232U_example.pdf",bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(verletLAPAction)
ax.plot(verletMEPAction)
