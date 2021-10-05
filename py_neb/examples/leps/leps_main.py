import sys
import os
import time
import pandas as pd

pyNebDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..//.."))
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

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
    
    def __call__(self,coordsArr): #(rab, x)
        """
        
        LEPs potential plus harmonic oscillator. TODO: add source
        
        Call this function with a numpy array of rab and x:
        
            xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
            zz = leps_plus_ho(xx,yy),
        
        and plot it as
        
            fig, ax = plt.subplots()
            ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")
    
        """
        if coordsArr.shape == (2,):
            coordsArr = coordsArr.reshape(1,2)
        
        rab = coordsArr[:,0]
        x = coordsArr[:,1]
        
        vOut = self.leps_pot(rab,self.rac-rab)
        vOut += 2*self.kc*(rab-(self.rac/2-x/self.c_ho))**2
        
        return vOut

#%%Setting up the potential on a grid
leps = LepsPot()

rab = np.arange(0,3.5,0.05)
x = np.arange(-3,3.05,0.05)

coordMeshTuple = np.meshgrid(rab,x)
flattenedCoords = np.array([c.flatten() for c in coordMeshTuple]).T

zz = leps(flattenedCoords).reshape(coordMeshTuple[0].shape)

#Shifting the ground state to be positive
eGS = zz.min()
potential = shift_func(leps,eGS)
zz = potential(flattenedCoords).reshape(coordMeshTuple[0].shape)

#Getting the initial and final points
minInds = find_local_minimum(zz)
coordsAtMinima = [c[minInds] for c in coordMeshTuple]

start = np.array([c[1] for c in coordsAtMinima])
end = np.array([c[0] for c in coordsAtMinima])

#%%Getting least action path (LAP) with Dijkstra's algorithm
dijkstra = Dijkstra(start,coordMeshTuple,zz,allowedEndpoints=end)
t0 = time.time()
_, pathArrDict, actionDict = dijkstra()
t1 = time.time()
path = pathArrDict[tuple(list(end))]
dijkstraAction = actionDict[tuple(list(end))]

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

verletLAP = VerletMinimization(lap,initialPath)
verletLAP.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
verletLAPAction = np.array([action(path,potential)[0] for path in verletLAP.allPts])

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

verletMEP = VerletMinimization(lap,initialPath)
verletMEP.fire(tStep,maxIters,useLocal=useLocal,fireParams={"dtMin":0.05})
verletMEPAction = np.array([action(path,potential)[0] for path in verletMEP.allPts])

#%%Printing action values
print("Action along path: %.3f" % dijkstraAction)
print("Action along Verlet LAP: %.3f" % verletLAPAction[-1])
print("Action along Verlet MEP: %.3f" % verletMEPAction[-1])

#%%Writing paths to text files
np.savetxt("Dijkstra_path.txt",path,delimiter=",")
np.savetxt("LAP.txt",verletLAP.allPts[-1],delimiter=",")
np.savetxt("MEP.txt",verletMEP.allPts[-1],delimiter=",")

#%%Plotting results
fig, ax = plt.subplots()
cf = ax.contourf(*coordMeshTuple,zz,levels=np.arange(0,50,1))
ax.contour(*coordMeshTuple,zz,levels=np.arange(0,6),colors=["black"])
ax.contour(*coordMeshTuple,zz,levels=np.arange(6,10),colors=["black"],\
           linestyles="dashed")
plt.colorbar(cf,ax=ax)

ax.scatter(*start,color="red",marker="x")
ax.scatter(*end,color="red",marker="^")

ax.set(xlabel="rAB",ylabel="x",title="LEPs + HO")

ax.plot(path[:,0],path[:,1],color="white",label="Dijkstra (%.3f)" % dijkstraAction)
ax.plot(verletLAP.allPts[-1,:,0],verletLAP.allPts[-1,:,1],marker=".",color="lime",\
        label="LAP (%.3f)" % verletLAPAction[-1])
ax.plot(verletMEP.allPts[-1,:,0],verletMEP.allPts[-1,:,1],marker=".",color="cyan",\
        label="MEP (%.3f)" % verletMEPAction[-1])

ax.legend()

fig.savefig("Leps_potential_example.pdf",bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(verletLAPAction)
ax.plot(verletMEPAction)
