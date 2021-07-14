import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters, morphology #For minimum finding
from scipy.signal import argrelextrema
import time

from scipy.integrate import solve_bvp
from shapely import geometry #Used in initializing the LNEB method on the gs contour

import pandas as pd
import h5py
import os
import sys

def find_local_minimum(arr):
    """
    Returns the indices corresponding to the local minimum values. Taken 
    directly from https://stackoverflow.com/a/3986876
    
    Parameters
    ----------
    arr : Numpy array
        A D-dimensional array.

    Returns
    -------
    minIndsOut : Tuple of numpy arrays
        D arrays of length k, for k minima found

    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood,\
                                        mode="nearest")==arr)
    
    background = (arr==0)
    #Not sure this is necessary - it doesn't seem to do much on the test
        #data I defined.
    eroded_background = morphology.binary_erosion(background,\
                                                  structure=neighborhood,\
                                                  border_value=1)
        
    detected_minima = local_min ^ eroded_background
    allMinInds = np.vstack(local_min.nonzero())
    
    minIndsOut = tuple([allMinInds[coordIter,:] for \
                        coordIter in range(allMinInds.shape[0])])
    
    return minIndsOut

def q(r,d,alpha,r0):
    return d/2*(3/2*np.exp(-2*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))

def j(r,d,alpha,r0):
    return d/4*(np.exp(-2*alpha*(r-r0)) - 6*np.exp(-alpha*(r-r0)))

def leps_pot(rab,rbc):
    a = 0.05
    b = 0.8
    c = 0.05
    dab = 4.746
    dbc = 4.746
    dac = 3.445
    r0 = 0.742
    alpha = 1.942
    
    rac = rab + rbc
    
    vOut = q(rab,dab,alpha,r0)/(1+a) + q(rbc,dbc,alpha,r0)/(1+b) + q(rac,dac,alpha,r0)/(1+c)
    
    jab = j(rab,dab,alpha,r0)
    jbc = j(rbc,dbc,alpha,r0)
    jac = j(rac,dac,alpha,r0)
    
    jTerm = jab**2/(1+a)**2+jbc**2/(1+b)**2+jac**2/(1+c)**2
    jTerm = jTerm - jab*jbc/((1+a)*(1+b)) - jbc*jac/((1+b)*(1+c)) - jab*jac/((1+a)*(1+c))
    
    vOut = vOut - np.sqrt(jTerm)
    
    return vOut

def leps_plus_ho(rab,x):
    """
    You should be able to just call this function with a numpy array of rab and x.
    
    For instance, I would call it as
    
        xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
        zz = leps_plus_ho(xx,yy),
    
    and plot it as
    
        fig, ax = plt.subplots()
        ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")

    """
    rac = 3.742
    vOut = leps_pot(rab,rac-rab)
    
    kc = 0.2025
    c = 1.154
    
    vOut += 2*kc*(rab-(rac/2-x/c))**2
    
    return vOut

class NudgedElasticBand():
    """
    This can be ignored for computing the potential.
    """
    def __init__(self,nPts=12,pot=leps_plus_ho,initialPoints=None,gsEneg=None,\
                 subtractEGS=False):
        #nPts includes both ends
        self.nPts = nPts
        self.nCoords = 2 #Variable in principle; unneeded here
        self.coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,2,0.05))
        self.zz = pot(*self.coordMeshTuple)
        self.potential = pot
        
        if initialPoints is None:
            self.initialPoints = self._initial_points()
        else:
            self.initialPoints = initialPoints
        
        if subtractEGS and (gsEneg is None):
            self.gsEneg = np.min(self.potential(*self.initialPoints))
        elif subtractEGS and (gsEneg is not None):
            self.gsEneg = gsEneg
        else:
            self.gsEneg = 0
        
    def _initial_points(self):
        minInds = find_local_minimum(self.zz)
        
        initialPoints = np.array([np.linspace(cmesh[minInds][0],cmesh[minInds][1],num=self.nPts) for \
                                  cmesh in self.coordMeshTuple])
        
        return initialPoints
    
    def _compute_tangents(self,currentPts,energies):
        tangents = np.zeros((self.nCoords,self.nPts))
        for ptIter in range(1,self.nPts-1): #Range selected to exclude endpoints
            tp = np.array([currentPts[cIter][ptIter+1] - \
                           currentPts[cIter][ptIter] for cIter in range(self.nCoords)])
            tm = np.array([currentPts[cIter][ptIter] - \
                           currentPts[cIter][ptIter-1] for cIter in range(self.nCoords)])
            dVMax = np.max(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
            dVMin = np.min(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
                
            if (energies[ptIter+1] > energies[ptIter]) and \
                (energies[ptIter] > energies[ptIter-1]):
                tangents[:,ptIter] = tp
            elif (energies[ptIter+1] < energies[ptIter]) and \
                (energies[ptIter] < energies[ptIter-1]):
                tangents[:,ptIter] = tm
            elif energies[ptIter+1] > energies[ptIter-1]:
                tangents[:,ptIter] = tp*dVMax + tm*dVMin
            #Paper gives this as just <, not <=. Probably won't come up...
            elif energies[ptIter+1] <= energies[ptIter-1]:
                tangents[:,ptIter] = tp*dVMin + tm*dVMax
                
            #Normalizing vectors
            tangents[:,ptIter] = tangents[:,ptIter]/np.sqrt(np.dot(tangents[:,ptIter],tangents[:,ptIter]))
        
        return tangents
    
    def _compute_true_force(self,currentPts,energies,tangents):
        eps = 10**(-6)
        
        gradEnegs = np.zeros((self.nCoords,self.nPts))
        for ptIter in range(1,self.nPts-1):
            thisPt = np.array([currentPts[cIter,ptIter] for cIter in range(self.nCoords)])
            enegAtPt = energies[ptIter]
            for cIter in range(self.nCoords):
                h = np.zeros(self.nCoords)
                h[cIter] = eps
                
                enegAtStep = self.potential(*(thisPt+h)) - self.gsEneg
                gradEnegs[cIter,ptIter] = (enegAtStep - enegAtPt)/eps
        
        #Was a typo here - didn't multiply by tangents (so added vector + scalar)
        trueForce = gradEnegs - np.array([np.dot(gradEnegs[:,ptIter],tangents[:,ptIter])\
                                          for ptIter in range(self.nPts)])*tangents
        
        return trueForce
    
    def _compute_broken_true_force(self,currentPts,energies,tangents):
        eps = 10**(-6)
        
        gradEnegs = np.zeros((self.nCoords,self.nPts))
        for ptIter in range(1,self.nPts-1):
            thisPt = np.array([currentPts[cIter,ptIter] for cIter in range(self.nCoords)])
            enegAtPt = energies[ptIter]
            for cIter in range(self.nCoords):
                h = np.zeros(self.nCoords)
                h[cIter] = eps
                
                enegAtStep = self.potential(*(thisPt+h)) - self.gsEneg
                gradEnegs[cIter,ptIter] = (enegAtStep - enegAtPt)/eps
        
        #Broken here b/c don't multiply by tangent vector
        trueForce = gradEnegs - np.array([np.dot(gradEnegs[:,ptIter],tangents[:,ptIter])\
                                          for ptIter in range(self.nPts)])
        
        return trueForce
    
    def _compute_parallel_force(self,currentPts,tangents,k):
        fOut = np.zeros((self.nCoords,self.nPts))
        
        for ptIter in range(1,self.nPts-1):
            rDiff = np.sqrt(np.dot(currentPts[:,ptIter+1]-currentPts[:,ptIter],\
                                   currentPts[:,ptIter+1]-currentPts[:,ptIter]))
            rDiff -= np.sqrt(np.dot(currentPts[:,ptIter]-currentPts[:,ptIter-1],\
                                    currentPts[:,ptIter]-currentPts[:,ptIter-1]))
            fOut[:,ptIter] = k*rDiff*tangents[:,ptIter]
        
        return fOut
    
    def compute_force(self,currentPts,k,useBrokenForce=False):
        energies = self.potential(*currentPts) - self.gsEneg
        
        tangents = self._compute_tangents(currentPts,energies)
        
        if useBrokenForce:
            trueForce = self._compute_broken_true_force(currentPts,energies,tangents)
        else:
            trueForce = self._compute_true_force(currentPts,energies,tangents)
        parallelForce = self._compute_parallel_force(currentPts,tangents,k=k)
        
        force = parallelForce - trueForce
        
        return force
    
    def action_integral(self,path):
        actOut = 0
        enegs = self.potential(*path) - self.gsEneg
        
        for ptIter in range(self.nPts-1):
            dist2 = 0
            for coordIter in range(self.nCoords):
                dist2 += (path[coordIter,ptIter+1] - path[coordIter,ptIter])**2
            actOut += enegs[ptIter]*np.sqrt(dist2)
        
        return actOut
    
    def verlet_minimization(self,maxIters=1000,tStep=0.05,k=50,useBrokenForce=False):
        #Idk if this k value makes sense. Changing it maybe doesn't impact the 
        #final path found - I thought it did, but now it doesn't appear to...
        
        allPts = np.zeros((maxIters+1,self.nCoords,self.nPts))
        allVelocities = np.zeros((maxIters+1,self.nCoords,self.nPts))
        allForces = np.zeros((maxIters+1,self.nCoords,self.nPts))
        
        allPts[0,:,:] = self.initialPoints
        
        for step in range(0,maxIters):
            force = self.compute_force(allPts[step,:,:],k,useBrokenForce=useBrokenForce)
            allForces[step+1,:,:] = force
            allPts[step+1,:,:] = allPts[step,:,:] + allVelocities[step,:,:]*tStep\
                + 1/2*force*tStep**2
                
            #Velocity update taken from "Classical and Quantum Dynamics in Condensed Phase Simulations",
            #page 397. Eric also updates with tStep * allForces[...]
            for ptIter in range(self.nPts):
                if np.dot(allVelocities[step,:,ptIter],allForces[step+1,:,ptIter])>0:
                    allVelocities[step+1,:,ptIter] = \
                        np.dot(allVelocities[step,:,ptIter],allForces[step+1,:,ptIter])*\
                            allForces[step+1,:,ptIter]/np.dot(allForces[step+1,:,ptIter],allForces[step+1,:,ptIter])
                else:
                    allVelocities[step+1,:,ptIter] = np.zeros(self.nCoords)\
                
        actions = np.array([self.action_integral(pts) for pts in allPts[:]])
        
        return allPts, allVelocities, allForces, actions
        
    def plot_pes(self,curves=[],labels=[],fName=None):
        fig, ax = plt.subplots()
        ax.contour(*self.coordMeshTuple,self.zz,np.arange(-10,70,1),colors="k")
        
        if curves: #Checks if list is nonempty
            for (cIter,curve) in enumerate(curves):
                if labels:
                    lab = labels[cIter]
                else:
                    lab = None
                ax.plot(curve[0,:],curve[1,:],ls="-",marker="o",label=lab)
                
        ax.set_xlabel(r"$r_{AB}$")
        ax.set_ylabel(r"$x$")
        
        if labels:
            ax.legend()
                
        if fName is not None:
            fig.savefig(fName+".pdf")
        
        return None
    
    def plot_action(self,actionValsList,titles=None,minLine=True,extraPt=None,ptName=None,fName=None):
        fig, ax = plt.subplots()
        if isinstance(actionValsList,np.ndarray):
            ax.plot(actionValsList)
        elif isinstance(actionValsList,list):
            for (lIter,l) in enumerate(actionValsList):
                if titles is not None:
                    label = titles[lIter]
                else:
                    label = None
                ax.plot(l,label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Action")
        
        ax.set_title("Action Integral in NEB Method")
        
        artist = []
        names = []
        
        if minLine:
            ax.axhline(actionValsList.min(),color="red")
            artist += [plt.Line2D((0,1),(0,0),color="red")]
            names += ["Min = %.3f" % actionValsList.min()]
        
        if extraPt is not None:
            ax.axhline(extraPt,color="green")
            artist += [plt.Line2D((0,1),(0,0),color="green")]
            names += [ptName+" = %.3f" % extraPt]
        
        if artist: #Checks if nonempty
            ax.legend(artist,names,loc="upper right")
            
        if titles is not None:
            ax.legend()
        
        if fName is not None:
            fig.savefig(fName+".pdf")
            
        return None
    
class LineIntegralNeb(NudgedElasticBand):
    def __init__(self,nPts=12,potential=leps_plus_ho,initialPoints=None,startMethod=0,\
                 gsEneg=None,subtractEGS=True):
        startMethodsDict = {0:self._initial_points,1:self._initial_on_contour}
        if startMethod not in startMethodsDict.keys():
            sys.exit("Error: initial points requested with method "+str(startMethod))
        
        #nPts includes both ends
        self.nPts = nPts
        self.nCoords = 2 #Variable in principle; unneeded here
        self.coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        self.zz = potential(*self.coordMeshTuple)
        
        if initialPoints is None:
            self.initialPoints = startMethodsDict[startMethod]()
        else:
            self.initialPoints = initialPoints
        
        if subtractEGS and (gsEneg is None):
            self.gsEneg = np.min(self.zz)
        elif subtractEGS and (gsEneg is not None):
            self.gsEneg = gsEneg
        else:
            self.gsEneg = 0
            
        self.potential = self._aux_pot(potential,self.gsEneg)
            
    def _initial_points(self):
        minInds = find_local_minimum(self.zz)
        
        initialPoints = np.array([np.linspace(cmesh[minInds][0],cmesh[minInds][1],num=self.nPts) for \
                                  cmesh in self.coordMeshTuple])
        
        return initialPoints
    
    def _initial_on_contour(self,debug=False):
        """
        Connects a straight line between the metastable state and the contour at
        the same energy.
        """
        minInds = find_local_minimum(self.zz)
        startInds = tuple([minInds[cIter][np.argmax(self.zz[minInds])] for\
                              cIter in range(self.nCoords)])
        metaEneg = self.zz[startInds]
        
        #Pulled from PES code. Doesn't generalize to higher dimensions, but not
        #an important issue rn.
        fig, ax = plt.subplots()
        ax.contourf(self.coordMeshTuple[0],self.coordMeshTuple[1],self.zz)
        contour = ax.contour(self.coordMeshTuple[0],self.coordMeshTuple[1],\
                             self.zz,levels=[metaEneg]).allsegs[0][0].T#Selects the actual curve
        if not debug:
            plt.close(fig)
            
        startPt = np.array([self.coordMeshTuple[tupInd][startInds] for tupInd in \
                            range(self.nCoords)])
        line = geometry.LineString(contour.T)
        
        approxFinalPt = np.array([1.5,1])
        point = geometry.Point(*approxFinalPt)
        finalPt = np.array(line.interpolate(line.project(point)))
        # print(point.distance(line))
            
        initialPoints = np.array([np.linspace(startPt[cInd],finalPt[cInd],num=self.nPts) for\
                                  cInd in range(self.nCoords)])
        
        return initialPoints
    
    def _aux_pot(self,potential,egs,tol=10**(-2)):
        #Tol is so that the potential at the ground state isn't exactly 0 
        #(causes issues in EL eqns; may cause issues here)
        def pot(x,y):
            return potential(x,y) - egs + tol
        
        return pot
    
    def _negative_gradient(self,points,energies):
        eps = 10**(-6)
        
        #The negative gradient of the function in the action, sqrt(2V(x))
        negGrads = np.zeros(points.shape)
        #Have to unravel this loop, at least, although all the points can be vectorized
        for coordIter in range(self.nCoords):
            steps = points.copy()
            steps[coordIter,:] += eps
            enegsAtSteps = self.potential(*steps)
            negGrads[coordIter,:] = (np.sqrt(2*enegsAtSteps) - np.sqrt(2*energies))/eps
        negGrads = -negGrads #Is negative b/c I want the actual force
        
        distVec = np.array([points[:,i] - points[:,i-1] for i in range(1,points.shape[1])]).T
        distScalar = np.array([np.linalg.norm(distVec[:,i]) for i in range(distVec.shape[1])])
        normedDistVec = np.array([distVec[:,i]/distScalar[i] for i in range(distVec.shape[1])]).T
        
        returnValue = np.zeros(points.shape)
        for i in range(1,self.nPts-1):#Endpoints are zeroed out
            #distScalar and normedDistVec are indexed starting one below the point index
            returnValue[:,i] = 1/np.sqrt(2*energies[i])*(distScalar[i-1]+distScalar[i])*negGrads[:,i]
            returnValue[:,i] -= (np.sqrt(2*energies[i])+np.sqrt(2*energies[i-1]))*normedDistVec[:,i-1]
            returnValue[:,i] += (np.sqrt(2*energies[i])+np.sqrt(2*energies[i+1]))*normedDistVec[:,i]
            
            returnValue[:,i] = returnValue[:,i]/2
        
        return returnValue, negGrads
    
    def _spring_force(self,points,tangents,k):
        diffArr = np.array([points[:,i+1] - points[:,i] for i in range(points.shape[1]-1)]).T
        diffScal = np.array([np.linalg.norm(diffArr[:,i]) for i in range(diffArr.shape[1])])
        
        returnValue = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            returnValue[:,i] = k*(diffScal[i] - diffScal[i-1])*tangents[:,i]
            
        returnValue[:,0] = k*diffArr[:,0]
        returnValue[:,-1] = k*diffArr[:,-1]
        
        return returnValue
    
    def compute_force(self,points,k,constraintEneg,kappa):
        energies = self.potential(*points)
        
        #Inherited from NudgedElasticBand class. Sloppy? Maybe. Is also (I think)
        #just normedDistVec from self._negative_gradient, so maybe pull from there
        tangents = self._compute_tangents(points,energies)
        negGrad, actualNegGrad = self._negative_gradient(points,energies)
        
        perpForce = negGrad - tangents*(np.array([np.dot(negGrad[:,i],tangents[:,i]) \
                                                 for i in range(points.shape[1])]))
        springForce = self._spring_force(points,tangents,k)
        
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            netForce[:,i] = perpForce[:,i] + springForce[:,i]
            
        normForce = actualNegGrad[:,0]/np.linalg.norm(actualNegGrad[:,0])
        netForce[:,0] = springForce[:,0] - \
            (np.dot(springForce[:,0],normForce)-kappa*(energies[0]-constraintEneg))*normForce
            
        normForce = actualNegGrad[:,-1]/np.linalg.norm(actualNegGrad[:,-1])
        netForce[:,-1] = springForce[:,-1] - \
            (np.dot(springForce[:,-1],normForce)-kappa*(energies[-1]-constraintEneg))*normForce
        
        return netForce
    
    def action_integral(self,path):
        actOut = 0
        enegs = self.potential(*path)
        
        for ptIter in range(1,self.nPts):
            dist2 = 0
            for coordIter in range(self.nCoords):
                dist2 += (path[coordIter,ptIter] - path[coordIter,ptIter-1])**2
            actOut += (np.sqrt(2*enegs[ptIter])+np.sqrt(2*enegs[ptIter-1]))*np.sqrt(dist2)
        
        return actOut/2
    
    def verlet_minimization(self,maxIters=1000,tStep=0.05,k=1,kappa=1):
        #For passing around when dumping to the output file
        self.maxIters = maxIters
        self.tStep = tStep
        self.k = k
        self.kappa = kappa
        
        constraintEneg = self.potential(*self.initialPoints)[0]
        self.constraintEneg = constraintEneg
        
        allPts = np.zeros((maxIters+1,self.nCoords,self.nPts))
        allVelocities = np.zeros((maxIters+1,self.nCoords,self.nPts))
        allForces = np.zeros((maxIters+1,self.nCoords,self.nPts))
        
        allPts[0,:,:] = self.initialPoints
        
        for step in range(0,maxIters):
            force = self.compute_force(allPts[step,:,:],k,constraintEneg,kappa)
            allForces[step+1,:,:] = force
            allPts[step+1,:,:] = allPts[step,:,:] + allVelocities[step,:,:]*tStep\
                + 1/2*force*tStep**2
            
            #Velocity update taken from "Classical and Quantum Dynamics in Condensed Phase Simulations",
            #page 397. Eric also updates with tStep * allForces[...]
            for ptIter in range(self.nPts):
                if np.dot(allVelocities[step,:,ptIter],allForces[step+1,:,ptIter])>0:
                    allVelocities[step+1,:,ptIter] = \
                        np.dot(allVelocities[step,:,ptIter],allForces[step+1,:,ptIter])*\
                            allForces[step+1,:,ptIter]/np.dot(allForces[step+1,:,ptIter],allForces[step+1,:,ptIter])
                else:
                    allVelocities[step+1,:,ptIter] = np.zeros(self.nCoords)\
                
        actions = np.array([self.action_integral(pts) for pts in allPts[:]])
        
        return allPts, allVelocities, allForces, actions
    
    def dump_to_hdf5(self,fname,optOutputs):
        allPts, allVelocities, allForces, actions = optOutputs
        
        if os.path.isfile(fname+".h5"):
            #Emergency output warning - in case I ever get careless
            print("Warning: output file "+fname+" exists; storing instead as "+fname+"1")
            fname = fname+"1"
        h5File = h5py.File(fname+".h5","w")
        h5File.create_dataset("allPts",data=allPts)
        h5File.create_dataset("allVelocities",data=allVelocities)
        h5File.create_dataset("allForces",data=allForces)
        h5File.create_dataset("actions",data=actions)
        
        h5File.create_group("Optimization_Params")
        h5File["Optimization_Params"].attrs.create("maxIters",self.maxIters)
        h5File["Optimization_Params"].attrs.create("tStep",self.tStep)
        h5File["Optimization_Params"].attrs.create("k",self.k)
        h5File["Optimization_Params"].attrs.create("kappa",self.kappa)
        h5File["Optimization_Params"].attrs.create("constraintEneg",self.constraintEneg)
        
        h5File.close()
        
        return None
    
class EulerLagrangeSolver:
    def __init__(self,nPts=12,potential=leps_plus_ho,initialPoints=None,gsEneg=None,\
                 subtractEGS=True):
        self.nPts = nPts
        self.coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,2,0.05))
        self.zz = potential(*self.coordMeshTuple)
        
        if initialPoints is None:
            self.initialPoints = self._initial_points()
        else:
            self.initialPoints = initialPoints
            
        if subtractEGS and (gsEneg is None):
            self.gsEneg = np.min(potential(*self.initialPoints))
        elif subtractEGS and (gsEneg is not None):
            self.gsEneg = gsEneg
        else:
            self.gsEneg = 0
            
        self.potential = self._aux_pot(potential,self.gsEneg)
            
    def _initial_points(self):
        minInds = find_local_minimum(self.zz)
        
        initialPoints = np.array([np.linspace(cmesh[minInds][0],cmesh[minInds][1],num=self.nPts) for \
                                  cmesh in self.coordMeshTuple])
        
        return initialPoints
    
    def _aux_pot(self,potential,egs,tol=10**(-4)):
        #Tol is so that the potential at the ground state isn't exactly 0 (have to
        #divide by V in the EL equations)
        def pot(x,y):
            return potential(x,y) - egs + tol
        
        return pot
        
    def _finite_difference(self,coords):
        eps = 10**(-6)
        
        potAtPt = self.potential(coords[0,:],coords[1,:])
        potAtStep = np.zeros((2,coords.shape[1]))
        for i in range(2):
            for coordIter in range(coords.shape[1]):
                locCoord = coords.copy()
                locCoord[i,coordIter] += eps
                potAtStep[i,coordIter] = self.potential(locCoord[0,coordIter],locCoord[1,coordIter])
        
        grad = (potAtStep - potAtPt)/eps
        
        return grad
        
    def rhs(self,indepVar,depVar):
        #depVar should be of shape (4,nPts) - 4 for (x0,x1,z0,z1); nPts for the size of the grid.
        #Here, zi = dxi/dt.
        funcOut = np.zeros(depVar.shape)
        #funcOut row order is (x0',x1',z0',z1') = (z0,z1,z0',z1')
        funcOut[0:2,:] = depVar[2:,:]
        
        #Not general here or anywhere else
        potEval = self.potential(depVar[0,:],depVar[1,:])
        potGrad = self._finite_difference(depVar[0:2,:])
        
        r = np.sqrt(depVar[2,:]**2 + depVar[3,:]**2)
        
        for gradIter in range(2):
            funcOut[gradIter+2,:] = 1/potEval*potGrad[gradIter,:]*r**2 - \
                depVar[gradIter+2,:]/potEval*(depVar[2,:]*potGrad[0,:]+depVar[3,:]*potGrad[1,:]) + \
                    depVar[gradIter+2,:]**2/r**2
        
        return funcOut
    
    def bc(self,ya,yb):
        #ya is right node, yb is left node. both of shape (4,), for (x0,x1,z0,z1)
        return np.array([ya[0] - self.initialPoints[0,0],ya[1] - self.initialPoints[-1,0],\
                         yb[0] - self.initialPoints[0,-1],yb[1] - self.initialPoints[-1,-1]])
    
    def plot_pes(self,curves=[],labels=[],fName=None):
        fig, ax = plt.subplots()
        
        ax.contour(*self.coordMeshTuple,self.zz,np.arange(-10,70,1),colors="k")
        
        if curves: #Checks if list is nonempty
            for (cIter,curve) in enumerate(curves):
                if labels:
                    lab = labels[cIter]
                else:
                    lab = None
                ax.plot(curve[0,:],curve[1,:],ls="-",marker="o",label=lab)
                
        ax.set_xlabel(r"$r_{AB}$")
        ax.set_ylabel(r"$x$")
        
        if labels:
            ax.legend()
                
        if fName is not None:
            fig.savefig(fName+".pdf")
        
        return None

def neb_main():
    nPts = 12
    
    neb = NudgedElasticBand(nPts=nPts,subtractEGS=True)
    allPts, allVelocities, allForces, actions = \
        neb.verlet_minimization(maxIters=200,tStep=0.1,k=10)
        
    neb.plot_action(actions)
    neb.plot_pes(curves=[allPts[0,:,:],allPts[np.argmin(actions),:,:],allPts[-1,:,:]],\
                  labels=["Initial","Minimum","Final"])
    return None

def el_main():
    test = EulerLagrangeSolver(nPts=nPts,initialPoints=allPts[-1,:,:])
    indepVar = np.linspace(0,1,nPts)
    depVar = np.vstack((test.initialPoints,0.1*np.ones(nPts),0.1*np.ones(nPts)))
    test.rhs(indepVar,depVar)
            
    testOut = solve_bvp(test.rhs,test.bc,indepVar,depVar)
    print(testOut.message)
    
    xSol, ySol, xGrad, yGrad = testOut.sol(indepVar)
    
    test.plot_pes(curves=[test.initialPoints,np.vstack((xSol,ySol)),test2.initialPoints,\
                          np.vstack((xSol2,ySol2))])
        
    return None

def lneb_main():
    lneb = LineIntegralNeb()
    allPts, allVelocities, allForces, actions = lneb.verlet_minimization(tStep=0.1,kappa=0,maxIters=2000)
    
    lneb.plot_action(actions,fName="LineIntegralNeb_Action")
    lneb.plot_pes(curves=[allPts[0],allPts[-1]],labels=["Initial","Final"],fName="LineIntegralNeb_Pes")
    
    return None

def compare_methods():
    lneb = LineIntegralNeb()
    lnebPts, _, _, lnebActions = \
        lneb.verlet_minimization(tStep=0.1,kappa=0,maxIters=2000)
        
    neb = NudgedElasticBand(subtractEGS=True)
    nebPts, _, _, nebActions = \
        neb.verlet_minimization(tStep=0.1,k=1,maxIters=2000)
        
    lneb.plot_pes(curves=[lnebPts[0],lnebPts[-1],nebPts[np.argmin(nebActions)],nebPts[-1]],\
                  labels=["Start","New NEB Minimum","Old NEB Minimum","Old NEB Converged"],\
                  fName="Path_Comparison")
        
    allActions = np.array([lneb.action_integral(nebPts[i]) for i in range(nebPts.shape[0])])
    
    lneb.plot_action([lnebActions,allActions],minLine=False,titles=["New NEB","Old NEB"],\
                     fName="Action_Comparison")
        
    return None

def lneb_on_contour():
    #Very clearly needs some work
    lneb = LineIntegralNeb(startMethod=1)
    allPts, allVelocities, allForces, actions = \
        lneb.verlet_minimization(tStep=0.1,kappa=10,maxIters=5000)
    
    lneb.plot_action(actions)
    lneb.plot_pes(curves=[allPts[0],allPts[-1]])
    
    fname = "Runs/"+time.ctime()
    lneb.dump_to_hdf5(fname,(allPts, allVelocities, allForces, actions))
    
    return None

def clean_dat():
    fileIn = "252U-PES.dat"
    
    arr = np.zeros((0,3))
    for (rIter,row) in enumerate(open(fileIn)):
        row = row.split(" ")
        row = list(filter(("").__ne__,row))
        row[2] = row[2].replace("\n","")
        
        if rIter == 0:
            heads = np.array(row[1:4])
        
        if rIter > 0:
            arr = np.vstack((arr,np.array(row,dtype=float)))
            
    xx, yy = (arr[:,0].reshape((351,-1)),arr[:,1].reshape((351,-1)))
    zz = arr[:,2].reshape(xx.shape)
    
    minInds = find_local_minimum(zz)
    gsInds = ([26],[5])
    eGS = zz[minInds][1]
    
    fig, ax = plt.subplots()
    cf = ax.contourf(xx,yy,zz-eGS)
    plt.colorbar(cf,ax=ax)
    
    ax.scatter(xx[gsInds],yy[gsInds],marker="x",color="red")
    sylvesterPts = (np.array([27,185]),np.array([0,16.2]))
    ax.scatter(sylvesterPts[0],sylvesterPts[1],marker="^",color="yellow")
    ax.set_ylim(np.min(yy),np.max(yy))
    
    h5File = h5py.File("252U_PES.h5","w")
    h5File.create_dataset("Q20",shape=xx.shape,data=xx)
    h5File.create_dataset("Q30",shape=xx.shape,data=yy)
    h5File.create_dataset("PES",shape=xx.shape,data=zz-eGS)
    
    h5File.attrs.create("GS_Loc",data=np.array([26,5]))
    h5File.attrs.create("GS_Eneg",data=eGS)
    
    h5File.close()
    
    return None

clean_dat()

