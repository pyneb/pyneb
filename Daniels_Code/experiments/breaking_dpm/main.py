import sys
import os
import time
import h5py
import pandas as pd

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *
np.seterr(all="raise")

x1 = np.array([0.,0.3,0.6,1])
x2 = np.array([0.,0.5,1])

coordMeshTuple = np.meshgrid(x1,x2)
zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
initialPoint = np.array([0.,0])
finalPoint = np.array([1.,1])

dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,\
                        allowedEndpoints=finalPoint,logLevel=0)
    
minIndsDict, minPathDict, distsDict = dp()

allPaths = np.array([[[0,0]]+[[0.3,x2[i]]]+[[0.6,x2[j]]]+[[1,1]] for i in range(3) \
                     for j in range(3)])
acts = np.array([dp.target_func(p,p[:,0]+2*p[:,1],None)[0] for p in allPaths])
minPathInd = np.argmin(acts)

fig, ax = plt.subplots()

denseMesh = np.meshgrid(x1,np.arange(-0.1,1.1,0.1))
ax.contourf(*denseMesh,denseMesh[0]+2*denseMesh[1],cmap="Spectral_r")
ax.plot(*minPathDict[tuple(finalPoint)].T,c="red",label="DPM (%.3f)"%distsDict[tuple(finalPoint)])
ax.plot(*allPaths[minPathInd].T,c="black",label="Actual\nShortest (%.3f)"%acts[minPathInd])

ax.set(xticks=x1,yticks=x2,ylim=(-0.1,1.1),xlabel="x",ylabel="y",title="V(x,y)=x+2y")

for y in x2:
    ax.plot([0.3,0.6],[y,y],color="blue",ls="--",marker=".")
    
lbl = "Shortest Path\nBetween Nodes"
ax.annotate(lbl,xy=(0.45,1),\
            xytext=(0.1,0.75),arrowprops={"arrowstyle":"-|>"})
ax.annotate(lbl,xy=(0.45,0.5),alpha=0.0,\
            xytext=(0.1,0.75),arrowprops={"arrowstyle":"-|>"})
ax.annotate(lbl,xy=(0.45,0),alpha=0.0,\
            xytext=(0.1,0.75),arrowprops={"arrowstyle":"-|>"})

ax.grid()
ax.legend(title="Action")

fig.savefig("dpm_counterexample.pdf")