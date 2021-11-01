#Appears to be common/best practice to import required packages in every file
#they are used in
import numpy as np
from scipy.ndimage import filters, morphology #For minimum finding

#For ND interpolation
# from scipy.interpolate import interpnd, RectBivariateSpline
import itertools

from scipy.integrate import solve_bvp

import h5py
import sys
import time
import warnings

from utilities import *
from fileio import *

"""
CONVENTIONS:
    -Paths should be of shape (nPoints, nDimensions)
    -Functions (e.g. a potential) that take in a single point should assume the
        first index of the array iterates over the points
    -Similarly, functions (e.g. the action) that take in many points should also
        assume the first index iterates over the points
"""
    
class LeastActionPath:
    """
    class documentation...?
    
    :Maintainer: Daniel
    """
    def __init__(self,potential,nPts,nDims,mass=None,endpointSpringForce=True,\
                 endpointHarmonicForce=True,target_func=TargetFunctions.action,\
                 target_func_grad=GradientApproximations().forward_action_grad,\
                 nebParams={},logLevel=1,loggerSettings={}):
        """
        asdf

        Parameters
        ----------
        potential : Function
            To be called as potential(path). Is passed to "target_func".
        endpointSpringForce : Bool or tuple of bools
            If a single bool, behavior is applied to both endpoints. If is a tuple
            of bools, the first stands for the index 0 on the path; the second stands
            for the index -1 on the path. TODO: possibly allow for a complicated
            function that returns a bool?
        nPts : Int
            Number of points on the band, including endpoints.
        nDims : Int
            Number of dimensions of the collective coordinates. For instance,
            when working with (Q20,Q30), nDims = 2.
        mass : Function, optional
            To be called as mass(path). Is passed to "target_func". If mass == None,
            the collective inertia is the identity matrix. The default is None.
        target_func : Function, optional
            The approximation of the action integral. Should take as arguments
            (path, potential, mass). Should return (action, potentialAtPath, massesAtPath).
            The default is action.
        target_func_grad : Function, optional
            Approximate derivative of the action integral with respect to every point.
            Should take as arguments 
                (path, potentialFunc, potentialOnPath, massFunc, massOnPath, target_func),
            where target_func is the action integral approximation. Should return 
            (gradOfAction, gradOfPes). The default is forward_action_grad.
        nebParams : Dict, optional
            Keyword arguments for the nudged elastic band (NEB) method. Controls
            the spring force and the harmonic oscillator potential. Default
            parameters are controlled by a dictionary in the __init__ method.
            The default is {}.

        Returns
        -------
        None.

        """
        #TODO: consider not having NEB parameters as a dictionary. Could be confusing...?
        defaultNebParams = {"k":10,"kappa":20,"constraintEneg":0}
        for key in defaultNebParams.keys():
            if key not in nebParams:
                nebParams[key] = defaultNebParams[key]
        
        #Not sure why I did things this way...
        for key in nebParams.keys():
            setattr(self,key,nebParams[key])
        #For compatibility with the logger
        self.nebParams = nebParams
            
        if isinstance(endpointSpringForce,bool):
            endpointSpringForce = 2*(endpointSpringForce,)
        if not isinstance(endpointSpringForce,tuple):
            raise ValueError("Unknown value "+str(endpointSpringForce)+\
                             " for endpointSpringForce")
                
        if isinstance(endpointHarmonicForce,bool):
            endpointHarmonicForce = 2*(endpointHarmonicForce,)
        if not isinstance(endpointHarmonicForce,tuple):
            raise ValueError("Unknown value "+str(endpointHarmonicForce)+\
                             " for endpointSpringForce")
        
        self.potential = potential
        self.mass = mass
        self.endpointSpringForce = endpointSpringForce
        self.endpointHarmonicForce = endpointHarmonicForce
        self.nPts = nPts
        self.nDims = nDims
        self.target_func = target_func
        self.target_func_grad = target_func_grad
        
        self.logger = ForceLogger(self,logLevel,loggerSettings,".lap")
    
    def _compute_tangents(self,points,energies):
        """
        Here for testing sphinx autodoc
        
        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        energies : TYPE
            DESCRIPTION.

        Returns
        -------
        tangents : TYPE
            DESCRIPTION.

        """
        tangents = np.zeros((self.nPts,self.nDims))
        
        #Range selected to exclude endpoints. Tangents on the endpoints do not
        #appear in the formulas.
        for ptIter in range(1,self.nPts-1):
            tp = points[ptIter+1] - points[ptIter]
            tm = points[ptIter] - points[ptIter-1]
            dVMax = np.max(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
            dVMin = np.min(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
                
            if (energies[ptIter+1] > energies[ptIter]) and \
                (energies[ptIter] > energies[ptIter-1]):
                tangents[ptIter] = tp
            elif (energies[ptIter+1] < energies[ptIter]) and \
                (energies[ptIter] < energies[ptIter-1]):
                tangents[ptIter] = tm
            elif energies[ptIter+1] > energies[ptIter-1]:
                tangents[ptIter] = tp*dVMax + tm*dVMin
            else:
                tangents[ptIter] = tp*dVMin + tm*dVMax
                
            #Normalizing vectors, without throwing errors about zero tangent vector
            if not np.array_equal(tangents[ptIter],np.zeros(self.nDims)):
                tangents[ptIter] = tangents[ptIter]/np.linalg.norm(tangents[ptIter])
        
        return tangents
    
    def _spring_force(self,points,tangents):
        """
        Spring force taken from https://doi.org/10.1063/1.5007180 eqns 20-22

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        tangents : TYPE
            DESCRIPTION.

        Returns
        -------
        springForce : TYPE
            DESCRIPTION.

        """
        springForce = np.zeros((self.nPts,self.nDims))
        for i in range(1,self.nPts-1):
            forwardDist = np.linalg.norm(points[i+1] - points[i])
            backwardsDist = np.linalg.norm(points[i] - points[i-1])
            springForce[i] = self.k*(forwardDist - backwardsDist)*tangents[i]
            
        if self.endpointSpringForce[0]:
            springForce[0] = self.k*(points[1] - points[0])
        
        if self.endpointSpringForce[1]:
            springForce[-1] = self.k*(points[self.nPts-2] - points[self.nPts-1])
        
        return springForce
    
    def compute_force(self,points):
        expectedShape = (self.nPts,self.nDims)
        if points.shape != expectedShape:
            if (points.T).shape == expectedShape:
                warnings.warn("Transposing points; (points.T).shape == expectedShape")
                points = points.T
            else:
                sys.exit("Err: points "+str(points)+\
                         " does not match expected shape in LeastActionPath")
        
        integVal, energies, masses = TargetFunctions.action(points,self.potential,self.mass)
        tangents = self._compute_tangents(points,energies)
        
        gradOfAction, gradOfPes = \
            self.target_func_grad(points,self.potential,energies,self.mass,masses,\
                                  self.target_func)
                
        negIntegGrad = -gradOfAction
        trueForce = -gradOfPes
        
        projection = np.array([np.dot(negIntegGrad[i],tangents[i]) \
                               for i in range(self.nPts)])
        parallelForce = np.array([projection[i]*tangents[i] for i in range(self.nPts)])
        perpForce = negIntegGrad - parallelForce

        springForce = self._spring_force(points,tangents)
        
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            netForce[i] = perpForce[i] + springForce[i]
        
        #TODO: add check if force is very small...?
        
        #Avoids throwing divide-by-zero errors, but also deals with points with
            #gradient within the finite-difference error from 0. Simplest example
            #is V(x,y) = x^2+y^2, at the origin. There, the gradient is the finite
            #difference value fdTol, in both directions, and so normalizing the
            #force artificially creates a force that should not be present
        if not np.allclose(trueForce[0],np.zeros(self.nDims),atol=fdTol):
            normForce = trueForce[0]/np.linalg.norm(trueForce[0])
        else:
            normForce = np.zeros(self.nDims)
        
        if self.endpointHarmonicForce[0]:
            netForce[0] = springForce[0] - (np.dot(springForce[0],normForce)-\
                                            self.kappa*(energies[0]-self.constraintEneg))*normForce
        else:
            netForce[0] = springForce[0]
        
        if not np.allclose(trueForce[-1],np.zeros(self.nDims),atol=fdTol):
            normForce = trueForce[-1]/np.linalg.norm(trueForce[-1])
        else:
            normForce = np.zeros(self.nDims)
        if self.endpointHarmonicForce[1]:
            netForce[-1] = springForce[-1] - (np.dot(springForce[-1],normForce)-\
                                              self.kappa*(energies[-1]-self.constraintEneg))*normForce
        else:
            netForce[-1] = springForce[-1]
        
        variablesDict = {"points":points,"tangents":tangents,"springForce":springForce,\
                         "netForce":netForce}
        self.logger.log(variablesDict)
        
        return netForce

class MinimumEnergyPath:
    """
    :Maintainer: Eric
    """
    def __init__(self,potential,nPts,nDims,endpointSpringForce=True,\
                 endpointHarmonicForce=True,auxFunc=None,\
                 target_func=TargetFunctions.mep_default,\
                 target_func_grad=potential_central_grad,nebParams={},\
                 logLevel=1,loggerSettings={}):
        """
        Parameters
        ----------
        potential : Function
            To be called as potential(path). This is the PES function. 
            Is passed to "target_func".
        endpointSpringForce : Bool or tuple of bools
            If a single bool, behavior is applied to both endpoints. If is a tuple
            of bools, the first stands for the index 0 on the path; the second stands
            for the index -1 on the path. TODO: possibly allow for a complicated
            function that returns a bool?
        nPts : Int
            Number of points on the band, including endpoints.
        nDims : Int
            Number of dimensions of the collective coordinates. For instance,
            when working with (Q20,Q30), nDims = 2.
        
        target_func : Function, optional
            The function to take the gradient of. Should take as arguments
            (path, potential, AuxFunc). AuxFunc is used to add an optional potential to
            modify the existing potential surface. This function should return 
            (potentialAtPath,AuxFuncAtPath). If no Aux function is needed, 
            input None for the argument and AuxFuncAtPath will be None. The default 
            is the PES potential (ie it just evaluates the potential).
        target_func_grad : Function, optional
            Approximate derivative of the target function evaluated at every point.
            Should take as arguments points,potential,auxFunc)
            where target_func is a potential target function . Should return 
            (gradOfPes, gradOfAux). If auxFunc is None, gradOfAux returns None
        nebParams : Dict, optional
            Keyword arguments for the nudged elastic band (NEB) method. Controls
            the spring force and the harmonic oscillator potential. Default
            parameters are controlled by a dictionary in the __init__ method.
            The default is {}.

        Returns
        -------
        None.

        """
        defaultNebParams = {"k":10,"kappa":20,"constraintEneg":0}
        for key in defaultNebParams.keys():
            if key not in nebParams:
                nebParams[key] = defaultNebParams[key]
        
        for key in nebParams.keys():
            setattr(self,key,nebParams[key])
        self.nebParams = nebParams
            
        if isinstance(endpointSpringForce,bool):
            endpointSpringForce = 2*(endpointSpringForce,)
        if not isinstance(endpointSpringForce,tuple):
            raise ValueError("Unknown value "+str(endpointSpringForce)+\
                             " for endpointSpringForce")
                
        if isinstance(endpointHarmonicForce,bool):
            endpointHarmonicForce = 2*(endpointHarmonicForce,)
        if not isinstance(endpointHarmonicForce,tuple):
            raise ValueError("Unknown value "+str(endpointHarmonicForce)+\
                             " for endpointSpringForce")
        
        self.potential = potential
        self.auxFunc = auxFunc
        self.endpointSpringForce = endpointSpringForce
        self.endpointHarmonicForce = endpointHarmonicForce
        self.nPts = nPts
        self.nDims = nDims
        self.target_func = target_func
        self.target_func_grad = target_func_grad
        
        self.logger = ForceLogger(self,logLevel,loggerSettings,".mep")
    
    def _compute_tangents(self,points,energies):
        """
        Here for testing sphinx autodoc
        
        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        energies : TYPE
            DESCRIPTION.

        Returns
        -------
        tangents : TYPE
            DESCRIPTION.

        """
        tangents = np.zeros((self.nPts,self.nDims))
        
        #Range selected to exclude endpoints. Tangents on the endpoints do not
        #appear in the formulas.
        for ptIter in range(1,self.nPts-1):
            tp = points[ptIter+1] - points[ptIter]
            tm = points[ptIter] - points[ptIter-1]
            dVMax = np.max(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
            dVMin = np.min(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
                
            if (energies[ptIter+1] > energies[ptIter]) and \
                (energies[ptIter] > energies[ptIter-1]):
                tangents[ptIter] = tp
            elif (energies[ptIter+1] < energies[ptIter]) and \
                (energies[ptIter] < energies[ptIter-1]):
                tangents[ptIter] = tm
            elif energies[ptIter+1] > energies[ptIter-1]:
                tangents[ptIter] = tp*dVMax + tm*dVMin
            else:
                tangents[ptIter] = tp*dVMin + tm*dVMax
                
            #Normalizing vectors, without throwing errors about zero tangent vector
            if not np.array_equal(tangents[ptIter],np.zeros(self.nDims)):
                tangents[ptIter] = tangents[ptIter]/np.linalg.norm(tangents[ptIter])
        
        return tangents
    
    def _spring_force(self,points,tangents):
        """
        Spring force taken from https://doi.org/10.1063/1.5007180 eqns 20-22

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        tangents : TYPE
            DESCRIPTION.

        Returns
        -------
        springForce : TYPE
            DESCRIPTION.

        """
        springForce = np.zeros((self.nPts,self.nDims))
        for i in range(1,self.nPts-1):
            forwardDist = np.linalg.norm(points[i+1] - points[i])
            backwardsDist = np.linalg.norm(points[i] - points[i-1])
            springForce[i] = self.k*(forwardDist - backwardsDist)*tangents[i]
            
        if self.endpointSpringForce[0]:
            springForce[0] = self.k*(points[1] - points[0])
        
        if self.endpointSpringForce[1]:
            springForce[-1] = self.k*(points[self.nPts-2] - points[self.nPts-1])
        
        return springForce
    
    def compute_force(self,points):
        expectedShape = (self.nPts,self.nDims)
        if points.shape != expectedShape:
            if (points.T).shape == expectedShape:
                warnings.warn("Transposing points; (points.T).shape == expectedShape")
                points = points.T
            else:
                sys.exit("Err: points "+str(points)+\
                         " does not match expected shape in MinimumEnergyPath")
        PESEnergies, auxEnergies = self.target_func(points,self.potential,self.auxFunc)
        tangents = self._compute_tangents(points,PESEnergies)
        gradOfPES, gradOfAux = \
            self.target_func_grad(points,self.potential,self.auxFunc)
        trueForce = -gradOfPES
        if gradOfAux is not None:
            negAuxGrad = -gradOfAux
            gradForce = trueForce + negAuxGrad
        else:
            gradForce = trueForce
        projection = np.array([np.dot(gradForce[i],tangents[i]) \
                               for i in range(self.nPts)])
        parallelForce = np.array([projection[i]*tangents[i] for i in range(self.nPts)])
        perpForce =  gradForce - parallelForce
        springForce = self._spring_force(points,tangents)
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        
        for i in range(1,self.nPts-1):
            netForce[i] = perpForce[i] + springForce[i]
        #Avoids throwing divide-by-zero errors, but also deals with points with
            #gradient within the finite-difference error from 0. Simplest example
            #is V(x,y) = x^2+y^2, at the origin. There, the gradient is the finite
            #difference value fdTol, in both directions, and so normalizing the
            #force artificially creates a force that should not be present
        if not np.allclose(gradForce[0],np.zeros(self.nDims),atol=fdTol):
            normForce = gradForce[0]/np.linalg.norm(gradForce[0])
        else:
            normForce = np.zeros(self.nDims)
        
        if self.endpointHarmonicForce[0]:
            netForce[0] = springForce[0] - (np.dot(springForce[0],normForce)-\
                                            self.kappa*(PESEnergies[0]-self.constraintEneg))*normForce
        else:
            netForce[0] = springForce[0]
        
        if not np.allclose(gradForce[-1],np.zeros(self.nDims),atol=fdTol):
            normForce = gradForce[-1]/np.linalg.norm(gradForce[-1])
        else:
            normForce = np.zeros(self.nDims)
        if self.endpointHarmonicForce[1]:
            netForce[-1] = springForce[-1] - (np.dot(springForce[-1],normForce)-\
                                              self.kappa*(PESEnergies[-1]-self.constraintEneg))*normForce
        else:
            netForce[-1] = springForce[-1]
            
        variablesDict = {"points":points,"tangents":tangents,"springForce":springForce,\
                         "netForce":netForce}
        self.logger.log(variablesDict)
            
        return netForce
    
#TODO: do we compute the action for all points after optimizing? or let someone else do that? 
class VerletMinimization:
    """
    :Maintainer: Daniel
    """
    def __init__(self,nebObj,initialPoints):
        """
        

        Parameters
        ----------
        nebObj : TYPE
            DESCRIPTION.
        initialPoints : TYPE
            DESCRIPTION.

        Raises
        ------
        AttributeError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.
        """
        #It'll probably do this automatically, but whatever
        if not hasattr(nebObj,"compute_force"):
            raise AttributeError("Object "+str(nebObj)+" has no attribute compute_force")
            
        self.nebObj = nebObj
        self.initialPoints = initialPoints
        self.nPts, self.nDims = initialPoints.shape
        
        if self.nPts != self.nebObj.nPts:
            raise ValueError("Obj "+str(self.nebObj)+" and initialPoints have "\
                             +"a different number of points")
                
        self.allPts = None
        self.allVelocities = None
        self.allForces = None
        
    def velocity_verlet(self,tStep,maxIters,dampingParameter=0):
        """
        Implements Algorithm 6 of https://doi.org/10.1021/acs.jctc.7b00360
        with optional damping force.
        
        TODO: that paper has many errors, esp. off-by-one errors. Could lead
        to issues. Consult http://dx.doi.org/10.1063/1.2841941 instead.
        TODO: modify to edit self.allPts and etc

        Parameters
        ----------
        tStep : TYPE
            DESCRIPTION.
        maxIters : TYPE
            DESCRIPTION.
        dampingParameter : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        allPts : TYPE
            DESCRIPTION.
        allVelocities : TYPE
            DESCRIPTION.
        allForces : TYPE
            DESCRIPTION.
        """
        #allPts is longer by 1 than the velocities/forces, because the last 
        #velocity/force computed should be used to update the points one 
        #last time (else that's computational time that's wasted)
        allPts = np.zeros((maxIters+2,self.nPts,self.nDims))
        allVelocities = np.zeros((maxIters+1,self.nPts,self.nDims))
        allForces = np.zeros((maxIters+1,self.nPts,self.nDims))
        
        vProj = np.zeros((self.nPts,self.nDims))
        
        allPts[0] = self.initialPoints
        allForces[0] = self.nebObj.compute_force(self.initialPoints)
        allVelocities[0] = tStep*allForces[0]
        allPts[1] = allPts[0] + allVelocities[0]*tStep + 0.5*allForces[0]*tStep**2
        
        for step in range(1,maxIters+1):
            allForces[step] = self.nebObj.compute_force(allPts[step])
            
            for ptIter in range(self.nPts):
                product = np.dot(allVelocities[step-1,ptIter],allForces[step,ptIter])
                if product > 0:
                    vProj[ptIter] = \
                        product*allForces[step,ptIter]/\
                            np.dot(allForces[step,ptIter],allForces[step,ptIter])
                else:
                    vProj[ptIter] = np.zeros(self.nDims)
                    
            #Damping term. Algorithm 6 uses allVelocities[step], but that hasn't
            #been computed yet. Note that this isn't applied to compute allPts[1].
            accel = allForces[step] - dampingParameter*allVelocities[step-1]                
            allVelocities[step] = vProj + tStep * accel
            
            allPts[step+1] = allPts[step] + allVelocities[step]*tStep + \
                0.5*accel*tStep**2
            
        return allPts, allVelocities, allForces
    
    def fire(self,tStep,maxIters,fireParams={},useLocal=False):
        """
        Wrapper for fast inertial relaxation engine.
        FIRE step taken from http://dx.doi.org/10.1103/PhysRevLett.97.170201
        
        Velocity update taken from 
        https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        
        TODO: consider making FIRE its own class, or allowing for attributes
        like fireParams and etc
        TODO: add maxmove parameter to prevent path exploding

        Parameters
        ----------
        tStep : TYPE
            DESCRIPTION.
        maxIters : TYPE
            DESCRIPTION.
        fireParams : TYPE, optional
            DESCRIPTION. The default is {}.
        useLocal : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.
        """
        
        defaultFireParams = \
            {"dtMax":10.,"dtMin":0.001,"nAccel":10,"fInc":1.1,"fAlpha":0.99,\
             "fDecel":0.5,"aStart":0.1}
            
        for key in fireParams.keys():
            if key not in defaultFireParams.keys():
                raise ValueError("Key "+key+" in fireParams not allowed")
                
        for key in defaultFireParams.keys():
            if key not in fireParams.keys():
                fireParams[key] = defaultFireParams[key]
                
        self.allPts = np.zeros((maxIters+2,self.nPts,self.nDims))
        self.allVelocities = np.zeros((maxIters+1,self.nPts,self.nDims))
        self.allForces = np.zeros((maxIters+1,self.nPts,self.nDims))
        
        self.allPts[0] = self.initialPoints
        self.allForces[0] = self.nebObj.compute_force(self.allPts[0])
        
        if useLocal:
            stepsSinceReset = np.zeros(self.nPts)
            tStepArr = np.zeros((maxIters+1,self.nPts))
            alphaArr = np.zeros((maxIters+1,self.nPts))
            stepsSinceReset = np.zeros(self.nPts)
        else:
            stepsSinceReset = 0
            tStepArr = np.zeros(maxIters+1)
            alphaArr = np.zeros(maxIters+1)
        
        tStepArr[0] = tStep
        alphaArr[0] = fireParams["aStart"]
        
        for step in range(1,maxIters+1):
            #TODO: check potential off-by-one indexing on tStep
            if useLocal:
                tStepArr,alphaArr,stepsSinceReset = \
                    self._local_fire_iter(step,tStepArr,alphaArr,stepsSinceReset,\
                                          fireParams)
            else:
                tStepArr,alphaArr,stepsSinceReset = \
                    self._global_fire_iter(step,tStepArr,alphaArr,stepsSinceReset,\
                                           fireParams)
        
        if useLocal:
            tStepFinal = tStepArr[-1].reshape((-1,1))
            self.allPts[-1] = self.allPts[-2] + tStepFinal*self.allVelocities[-1] + \
                0.5*self.allForces[-1]*tStepFinal**2
        else:
            self.allPts[-1] = self.allPts[-2] + tStepArr[-1]*self.allVelocities[-1] + \
                0.5*self.allForces[-1]*tStepArr[-1]**2
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _local_fire_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        tStepPrev = tStepArr[step-1].reshape((-1,1)) #For multiplication below
        shift = tStepPrev*self.allVelocities[step-1] + \
                0.5*self.allForces[step-1]*tStepPrev**2
        maxmove = 1.0
        for ptIter in range(self.nPts):
            if(np.linalg.norm(shift[ptIter])>maxmove):
                shift[ptIter] = maxmove * shift[ptIter]/np.linalg.norm(shift[ptIter])
        self.allPts[step] = self.allPts[step-1] + shift
        self.allForces[step] = self.nebObj.compute_force(self.allPts[step])

        #What the Wikipedia article on velocity Verlet uses
        self.allVelocities[step] = \
            0.5*tStepPrev*(self.allForces[step]+self.allForces[step-1])
        
        for ptIter in range(self.nPts):
            alpha = alphaArr[step-1,ptIter]
            
            product = np.dot(self.allVelocities[step-1,ptIter],self.allForces[step,ptIter])
            if product > 0:
                vMag = np.linalg.norm(self.allVelocities[step-1,ptIter])
                fHat = self.allForces[step,ptIter]/np.linalg.norm(self.allForces[step,ptIter])
                self.allVelocities[step,ptIter] += (1-alpha)*self.allVelocities[step-1,ptIter] + \
                    alpha*vMag*fHat
                
                if stepsSinceReset[ptIter] > fireParams["nAccel"]:
                    tStepArr[step,ptIter] = \
                        min(tStepArr[step-1,ptIter]*fireParams["fInc"],fireParams["dtMax"])
                    alphaArr[step,ptIter] = alpha*fireParams["fAlpha"]
                else:
                    tStepArr[step,ptIter] = tStepArr[step-1,ptIter]
                
                stepsSinceReset[ptIter] += 1
            else:
                tStepArr[step,ptIter] = \
                    max(tStepArr[step-1,ptIter]*fireParams["fDecel"],fireParams["dtMin"])
                alphaArr[step,ptIter] = fireParams["aStart"]
                stepsSinceReset[ptIter] = 0
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _global_fire_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        self.allPts[step] = self.allPts[step-1] + \
            tStepArr[step-1]*self.allVelocities[step-1] + \
            0.5*self.allForces[step-1]*tStepArr[step-1]**2
        self.allForces[step] = self.nebObj.compute_force(self.allPts[step])
        
        #Doesn't seem to make a difference
        #What the Wikipedia article on velocity Verlet uses
        self.allVelocities[step] = \
            0.5*tStepArr[step-1]*(self.allForces[step]+self.allForces[step-1])
        #What Eric uses
        # self.allVelocities[step] = tStepArr[step-1]*self.allForces[step]
        
        for ptIter in range(self.nPts):
            alpha = alphaArr[step-1]
            
            product = np.dot(self.allVelocities[step-1,ptIter],self.allForces[step,ptIter])
            if product > 0:
                vMag = np.linalg.norm(self.allVelocities[step-1,ptIter])
                fHat = self.allForces[step,ptIter]/np.linalg.norm(self.allForces[step,ptIter])
                vp = (1-alpha)*self.allVelocities[step-1,ptIter] + alpha*vMag*fHat
                self.allVelocities[step,ptIter] += vp
                
                if stepsSinceReset > fireParams["nAccel"]:
                    tStepArr[step] = min(tStepArr[step-1]*fireParams["fInc"],fireParams["dtMax"])
                    alphaArr[step] = alpha*fireParams["fAlpha"]
                
                stepsSinceReset += 1
            else:
                tStepArr[step] = max(tStepArr[step-1]*fireParams["fDecel"],fireParams["dtMin"])
                # self.allVelocities[step,ptIter] = np.zeros(self.nDims)
                alphaArr[step] = fireParams["aStart"]
                stepsSinceReset = 0
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _check_early_stop(self):
        
        return None
    
class EulerLagrangeSolver:
    """
    :Maintainer: Daniel
    """
    def __init__(self,initialPath,eneg_func,mass_func=None,grad_approx=midpoint_grad):
        self.initialPath = initialPath
        self.eneg_func = eneg_func
        self.mass_func = mass_func
        self.grad_approx = grad_approx
        
        self.nPts, self.nDims = initialPath.shape
        
    def _solve_id_inertia(self):
        def el(t,z):
            #z is the dependent variable. It is (x1,..., xD, x1',... xD').
            #For Nt images, t.shape == (Nt,), and z.shape == (2D,Nt).
            zOut = np.zeros(z.shape)
            zOut[:self.nDims] = z[self.nDims:]
            
            enegs = self.eneg_func(z[:self.nDims].T)
            enegGrad = self.grad_approx(self.eneg_func,z[:self.nDims].T)
            
            nPtsLoc = z.shape[1]
            for ptIter in range(nPtsLoc):
                v = z[self.nDims:,ptIter]
                term1 = 1/(2*enegs[ptIter])*enegGrad[ptIter] * np.dot(v,v)
                
                term2 = -1/(2*enegs[ptIter])*v*np.dot(enegGrad[ptIter],v)
                
                zOut[self.nDims:,ptIter] = term1 + term2
            
            return zOut
        
        def bc(z0,z1):
            leftPtCond = np.array([z0[i] - self.initialPath[0,i] for \
                                   i in range(self.nDims)])
            rightPtCond = np.array([z1[i] - self.initialPath[-1,i] for \
                                    i in range(self.nDims)])
            return np.concatenate((leftPtCond,rightPtCond))
        
        initialGuess = np.vstack((self.initialPath.T,np.zeros(self.initialPath.T.shape)))
        
        t = np.linspace(0.,1,self.nPts)
        sol = solve_bvp(el,bc,t,initialGuess)
        
        return sol
        
    def solve(self):
        if self.mass_func is None:
            sol = self._solve_id_inertia()
            
        return sol
    
class EulerLagrangeVerification:
    """
    :Maintainer: Daniel
    """
    def __init__(self,path,enegOnPath,eneg_func,massOnPath=None,mass_func=None,\
                 grad_approx=midpoint_grad):
        #TODO: major issue here when points aren't evenly spaced along the arc-length
        #of the path. For instance, a straight line of 3 nodes, on a V = constant
        #PES, will not pass this if the nodes aren't spaced evenly
        self.nPts, self.nDims = path.shape
        self.ds = 1/(self.nPts-1)
        
        massShape = (self.nPts,self.nDims,self.nDims)
        if massOnPath is None:
            self.massIsIdentity = True
            massOnPath = np.full(massShape,np.identity(self.nDims))
        else:
            self.massIsIdentity = False
            
        if enegOnPath.shape != (self.nPts,):
            raise ValueError("Invalid shape for enegOnPath. Expected "+str((self.nPts,))+\
                             ", received "+str(enegOnPath.shape))
        if massOnPath.shape != massShape:
            raise ValueError("Invalid shape for massOnPath. Expected "+str(massShape)+\
                             ", received "+str(massOnPath.shape))
                
        #Note that these are padded, for simplified indexing
        self.xDot = np.zeros((self.nPts,self.nDims))
        self.xDot[1:] = np.array([path[i]-path[i-1] for i in range(1,self.nPts)])/self.ds
        self.xDotDot = np.zeros((self.nPts,self.nDims))
        self.xDotDot[1:-1] = \
            np.array([path[i+1]-2*path[i]+path[i-1] for i in range(1,self.nPts-1)])/self.ds**2
            
        self.path = path
        self.enegOnPath = enegOnPath
        self.eneg_func = eneg_func
        self.massOnPath = massOnPath
        self.mass_func = mass_func
        self.grad_approx = grad_approx
    
    def _compare_lagrangian_id_inertia(self):
        enegGrad = self.grad_approx(self.eneg_func,self.path)
        
        lhs = np.zeros((self.nPts,self.nDims))
        rhs = np.zeros((self.nPts,self.nDims))
        
        for ptIter in range(1,self.nPts):
            lhs[ptIter] = 2*self.enegOnPath[ptIter]*self.xDotDot[ptIter]
            
            term1 = enegGrad[ptIter] * np.dot(self.xDot[ptIter],self.xDot[ptIter])
            term2 = -self.xDot[ptIter] * np.dot(enegGrad[ptIter],self.xDot[ptIter])
            
            rhs[ptIter] = term1 + term2
            
        diff = rhs - lhs
        return diff[1:-1] #Removing padding
    
    def _compare_lagrangian_var_inertia(self):
        enegGrad = self.grad_approx(self.eneg_func,self.path)
        
        lhs = np.zeros((self.nPts,self.nDims))
        rhs = np.zeros((self.nPts,self.nDims))
        
        #TODO: fill in eqns here
            
        diff = rhs - lhs
        
        return diff[1:-1] #Removing excess padding
    
    def compare_lagrangian(self):
        if self.massIsIdentity:
            elDiff = self._compare_lagrangian_id_inertia()
        else:
            elDiff = self._compare_lagrangian_var_inertia()
            
        
        return None
    
    def compare_lagrangian_squared(self):
        
        return None
    
class Dijkstra:
    """
    :Maintainer: Daniel
    """
    def __init__(self,initialPoint,coordMeshTuple,potArr,inertArr=None,\
                 target_func=TargetFunctions.action,allowedEndpoints=None,\
                 trimVals=[10**(-4),None],logLevel=1):
        """
        Some indexing is done to deal with the default shape of np.meshgrid.
        For D dimensions, the output is of shape (N2,N1,N3,...,ND), while the
        way indices are generated expects a shape of (N1,...,ND). So, I swap
        the first two indices by hand. See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        #TODO: error handling (try getting an index)(?)
        
        Note that indexing for Dijkstra *internal* functions are done in the
        order (N2,N1,N3,...), for simplicity. The indexing that is returned
        by self.__call__ is kept in this order by default.
        
        Note that the *value* of the array at a certain index is the same
        regardless of the sort order of the indices, provided that the index
        order matches that used when creating np.meshgrid

        Parameters
        ----------
        initialPoint : TYPE
            DESCRIPTION.
        coordMeshTuple : TYPE
            DESCRIPTION.
        potArr : TYPE
            DESCRIPTION.
        inertArr : TYPE, optional
            DESCRIPTION. The default is None.
        target_func : TYPE, optional
            DESCRIPTION. The default is action.
        allowedEndpoints : TYPE, optional
            DESCRIPTION. The default is None.
        trimVals : TYPE, optional
            DESCRIPTION. The default is [10**(-4),None].

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.initialPoint = initialPoint
        self.coordMeshTuple = coordMeshTuple
        self.uniqueCoords = [np.unique(c) for c in self.coordMeshTuple]
        
        expectedShape = np.array([len(c) for c in self.uniqueCoords])
        expectedShape[[1,0]] = expectedShape[[0,1]]
        expectedShape = tuple(expectedShape)
        
        tempCMesh = []
        for c in self.coordMeshTuple:
            if c.shape != expectedShape:
                cNew = np.swapaxes(c,0,1)
                if cNew.shape == expectedShape:
                    tempCMesh.append(cNew)
                else:
                    raise ValueError("coordMeshTuple has wrong dimensions somehow")
        if tempCMesh:
            self.coordMeshTuple = tuple(tempCMesh)
        
        self.target_func = target_func
        
        self.nDims = len(coordMeshTuple)
        
        if potArr.shape == expectedShape:
            self.potArr = potArr
        else:
            potNew = np.swapaxes(potArr,0,1)
            if potNew.shape == expectedShape:
                self.potArr = potNew
            else:
                raise ValueError("potArr.shape is "+str(potArr.shape)+\
                                 "; required shape is "+str(expectedShape)+\
                                 " (or with swapped first two indices)")
        #TODO: apply error checking above to inertArr
        if inertArr is not None:
            inertArrRequiredShape = self.potArr.shape + 2*(self.nDims,)
            if inertArr.shape != inertArrRequiredShape:
                raise ValueError("inertArr.shape is "+str(inertArr.shape)+\
                                 "; required shape is "+inertArrRequiredShape)
            self.inertArr = inertArr
        else:
            #Simplifies things in self._construct_path_dict if I set this to the 
            #identity here
            self.inertArr = np.full(self.potArr.shape+2*(self.nDims,),np.identity(self.nDims))
        
        if allowedEndpoints is None:
            self.allowedEndpoints, self.endpointIndices \
                = SurfaceUtils.find_endpoints_on_grid(self.coordMeshTuple,self.potArr)
        else:
            self.allowedEndpoints = allowedEndpoints
            self.endpointIndices, _ = \
                SurfaceUtils.round_points_to_grid(self.coordMeshTuple,allowedEndpoints)
        
        if self.allowedEndpoints.shape == (self.nDims,):
            self.allowedEndpoints = self.allowedEndpoints.reshape((1,self.nDims))
        
        if self.allowedEndpoints.shape[1] != self.nDims:
            raise ValueError("self.allowedEndpoints.shape == "+\
                             str(self.allowedEndpoints.shape)+"; dimension 1 must be "\
                             +str(self.nDims))
        if self.endpointIndices.shape[1] != self.nDims:
            raise ValueError("self.endpointIndices.shape == "+\
                             str(self.endpointIndices.shape)+"; dimension 1 must be "\
                             +str(self.nDims))
        
        #Flipping (N1,N2) -> (N2,N1) here. Expect all indices everywhere are
        #handled in the normal order (N1,N2,...)
        self.endpointIndices[:,[1,0]] = self.endpointIndices[:,[0,1]]
        self.endpointIndices = [tuple(row) for row in self.endpointIndices]
        
        #Clip the potential to the min/max. Done after finding possible endpoints.
        self.trimVals = trimVals
        if self.trimVals != [None,None]:
            self.potArr = self.potArr.clip(self.trimVals[0],self.trimVals[1])
        
        #Getting indices for self.initialPoint
        self.initialInds = np.zeros(self.nDims,dtype=int)
        for dimIter in range(self.nDims):
            #More nuisances with floating-point precision. Should maybe find
            #source of the issue, but this works for the basic test case.
            self.initialInds[dimIter] = \
                np.argwhere(np.isclose(self.uniqueCoords[dimIter],self.initialPoint[dimIter]))
        
        #Index swapping like mentioned above. Now, self.initialInds[0] <= Ny,
        #and self.initialInds[1] <= Nx
        self.initialInds[[1,0]] = self.initialInds[[0,1]]
        self.initialInds = tuple(self.initialInds)
        
        self.djkLogger = DijkstraLogger(self,logLevel=logLevel)
    
    def _construct_path_dict(self):
        """
        Uses Dijkstra's algorithm to determine the previous node visited
        for every node in the PES. See e.g. 
        https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

        TODO: allow for non-grid PES, like if we trimmed off high-energy regions
        Maybe fill to a grid, and set those guys to infinite energy so they're
        never selected? Simplest to have a list of indices that are not allowed,
        and set the mask to "visited" for those points, so that they are never
        even considered.
        
        Returns
        -------
        None.

        """
        t0 = time.time()
        
        suggestedMaxSize = 50000
        if self.potArr.size >= suggestedMaxSize:
            warnings.warn("Number of nodes is "+str(self.potArr.size)+\
                          "; recommended maximum method to finish is "\
                          +str(suggestedMaxSize))
        
        #Use a masked array to both track the distance and the visited values
        tentativeDistance = \
            np.ma.masked_array(np.inf*np.ones(self.potArr.shape),np.zeros(self.potArr.shape))
        
        #For current indices, to get to a neighbor, subtract one tuple from
        #relativeNeighborInds
        relativeNeighborInds = list(itertools.product([-1,0,1],repeat=self.nDims))
        relativeNeighborInds.remove(self.nDims*(0,))
        
        currentInds = self.initialInds
        tentativeDistance[currentInds] = 0
        
        neighborsVisitDict = {}
        endpointIndsList = self.endpointIndices.copy() #May need indices later
        
        #Index swapping like mentioned in self.__init__
        maxInds = np.array([len(c) for c in self.uniqueCoords])
        maxInds[[1,0]] = maxInds[[0,1]]
        
        #Ends when all endpoints have been reached, or it has iterated over every
        #node. Latter should not be possible; is a check in case something goes wrong.
        for i in range(self.potArr.size):
            neighborInds = np.array(currentInds) - relativeNeighborInds
            
            #Removing indices that take us off-grid. See e.g.
            #https://stackoverflow.com/a/20528566
            isNegativeBool = [neighborInds[:,i] < 0 for i in range(self.nDims)]
            neighborInds = \
                neighborInds[np.logical_not(np.logical_or.reduce(isNegativeBool))]
            isTooBigBool = [neighborInds[:,i] >= maxInds[i] for i in range(self.nDims)]
            neighborInds = \
                neighborInds[np.logical_not(np.logical_or.reduce(isTooBigBool))]
            
            #For feeding into self.target_func
            coords = np.zeros((2,self.nDims))
            coords[0] = np.array([c[currentInds] for c in self.coordMeshTuple])
            
            enegs = np.zeros(2)
            enegs[0] = self.potArr[currentInds]
            
            masses = np.zeros((2,)+2*(self.nDims,))
            masses[0] = self.inertArr[currentInds]
            
            for (neighIter, neighbor) in enumerate(neighborInds):
                n = tuple(neighbor)
                coords[1] = [c[n] for c in self.coordMeshTuple]
                enegs[1] = self.potArr[n]
                masses[1] = self.inertArr[n]
                
                #self.target_func returns the action (distance), plus energies and masses
                distThroughCurrent = tentativeDistance[currentInds] + \
                    self.target_func(coords,enegs,masses)[0]
                if distThroughCurrent < tentativeDistance[n]:
                    tentativeDistance[n] = distThroughCurrent
                    neighborsVisitDict[n] = currentInds
            
            tentativeDistance.mask[currentInds] = True
            
            #For checking when we want to stop
            try:
                endpointIndsList.remove(currentInds)
            except:
                pass
            
            currentInds = np.unravel_index(np.argmin(tentativeDistance),\
                                           tentativeDistance.shape)
            
            #Will exit early if all of the endpoints have been visited
            if not endpointIndsList:
                break
            
        t1 = time.time()
        runTime = t1 - t0
            
        var = (tentativeDistance,neighborsVisitDict,endpointIndsList,runTime)
        nms = ("tentativeDistance","neighborsVisitDict","endpointIndsList","runTime")
        
        self.djkLogger.log(var,nms)
        
        return tentativeDistance, neighborsVisitDict, endpointIndsList
    
    def _get_paths(self,neighborsVisitDict):
        allPathsIndsDict = {}
        for endptInds in self.endpointIndices:
            path = []
            step = endptInds
            while step != self.initialInds:
                path.append(step)
                step = neighborsVisitDict[step]
            path.append(self.initialInds)
            path.reverse()
            
            allPathsIndsDict[endptInds] = path
        
        var = (allPathsIndsDict,)
        nms = ("allPathsIndsDict",)
        
        self.djkLogger.log(var,nms)
        
        return allPathsIndsDict
    
    def __call__(self,returnAll=False):
        """
        

        Parameters
        ----------
        

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        tentativeDistance, neighborsVisitDict, endpointIndsList = \
            self._construct_path_dict()
        
        #Warns if any endpoint isn't visited
        if endpointIndsList:
            warnings.warn("Endpoint indices\n"+str(endpointIndsList)+\
                          "\nnot visited")
        pathIndsDict = self._get_paths(neighborsVisitDict)
        
        pathIndsDictRet = {} #Returns with the keys equal to the final point, not the index
        pathArrDict = {}
        distanceDict = {}
        for finalInds in pathIndsDict.keys():
            finalPt = np.array([c[finalInds] for c in self.coordMeshTuple])
            actualPath = np.zeros((0,self.nDims))
            for pathInd in pathIndsDict[finalInds]:
                toAppend = \
                    np.array([c[pathInd] for c in self.coordMeshTuple]).reshape((1,-1))
                actualPath = np.append(actualPath,toAppend,axis=0)
            
            pathIndsDictRet[tuple(finalPt.tolist())] = pathIndsDict[finalInds]
            pathArrDict[tuple(finalPt.tolist())] = actualPath
            distanceDict[tuple(finalPt.tolist())] = tentativeDistance.data[finalInds]
            
        #Don't log pathIndsDictRet, as it's logged under pathIndsDict
        #Don't log distanceDict, as it's logged under tentativeDistance
        var = (pathArrDict,)
        nms = ("pathArrDict",)
        self.djkLogger.log(var,nms)
        
        if returnAll:
            return pathIndsDictRet, pathArrDict, distanceDict
        else:
            endptOut = self.minimum_endpoint(distanceDict)
            return pathIndsDictRet[endptOut], pathArrDict[endptOut], distanceDict[endptOut]
    
    def minimum_endpoint(self,distanceDict):
        minDist = np.inf
        for (endpt, dist) in distanceDict.items():
            if dist < minDist:
                endptOut = endpt
                minDist = dist
        self.djkLogger.log((endptOut,),("endptOut",))
        return endptOut
