#Appears to be common/best practice to import required packages in every file
#they are used in
import numpy as np

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
        if not isinstance(endpointSpringForce,(tuple,list)):
            raise ValueError("Unknown value "+str(endpointSpringForce)+\
                             " for endpointSpringForce")
                
        if isinstance(endpointHarmonicForce,bool):
            endpointHarmonicForce = 2*(endpointHarmonicForce,)
        if not isinstance(endpointHarmonicForce,(tuple,list)):
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
        
        integVal, energies, masses = self.target_func(points,self.potential,self.mass)
        tangents = self._compute_tangents(points,energies)
        
        if self.logger.logLevel == 2:
            print("Action: ",integVal)
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
        
        """
        allPts is longer by 1 than the velocities/forces, because the last 
        velocity/force computed should be used to update the points one 
        last time (else that's computational time that's wasted)
        """
        self.allPts = np.zeros((maxIters+2,self.nPts,self.nDims))
        self.allVelocities = np.zeros((maxIters+1,self.nPts,self.nDims))
        self.allForces = np.zeros((maxIters+1,self.nPts,self.nDims))

        self.allPts[0] = self.initialPoints
        self.allForces[0] = self.nebObj.compute_force(self.allPts[0])
        self.allVelocities[0] = tStep*self.allForces[0]
        self.allPts[1] = self.allPts[0] + \
            self.allVelocities[0]*tStep + 0.5*self.allForces[0]*tStep**2
         
        vProj = np.zeros((self.nPts,self.nDims))
        
        t0 = time.time()
        try:
            for step in range(1,maxIters+1):
                self.allForces[step] = self.nebObj.compute_force(self.allPts[step])
                
                for ptIter in range(self.nPts):
                    product = np.dot(self.allVelocities[step-1,ptIter],self.allForces[step,ptIter])
                    if product > 0:
                        vProj[ptIter] = \
                            product*self.allForces[step,ptIter]/\
                                np.dot(self.allForces[step,ptIter],self.allForces[step,ptIter])
                    else:
                        vProj[ptIter] = np.zeros(self.nDims)
                        
                #Damping term. Algorithm 6 uses allVelocities[step], but that hasn't
                #been computed yet. Note that this isn't applied to compute allPts[1].
                accel = self.allForces[step] - dampingParameter*self.allVelocities[step-1]                
                self.allVelocities[step] = vProj + tStep * accel
                
                self.allPts[step+1] = self.allPts[step] + self.allVelocities[step]*tStep + \
                    0.5*accel*tStep**2
        finally:
            t1 = time.time()
            self.nebObj.logger.flush()
            self.nebObj.logger.write_runtime(t1-t0)
            
        return None
    
    def fire(self,tStep,maxIters,fireParams={},useLocal=True,earlyStop=True,
             earlyStopParams={},earlyAbort=False,earlyAbortParams={}):
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
             "fDecel":0.5,"aStart":0.1,"maxmove":np.full(self.nDims,1.0)}
            
        for key in fireParams.keys():
            if key not in defaultFireParams.keys():
                raise ValueError("Key "+key+" in fireParams not allowed")
                
        for key in defaultFireParams.keys():
            if key not in fireParams.keys():
                fireParams[key] = defaultFireParams[key]
                
        defaultStopParams = {"startCheckIter":300,"nStabIters":50,"checkFreq":10,\
                             "stabPerc":0.002}
        
        for key in earlyStopParams.keys():
            if key not in defaultStopParams.keys():
                raise ValueError("Key "+key+" in earlyStopParams not allowed")
                
        for key in defaultStopParams.keys():
            if key not in earlyStopParams.keys():
                earlyStopParams[key] = defaultStopParams[key]
                
        defaultAbortParams = {"startCheckIter":20,"nStabIters":10,"checkFreq":10,\
                              "variance":1}
            
        for key in earlyAbortParams.keys():
            if key not in defaultAbortParams.keys():
                raise ValueError("Key "+key+" in earlyStopParams not allowed")
                
        for key in defaultAbortParams.keys():
            if key not in earlyAbortParams.keys():
                earlyAbortParams[key] = defaultAbortParams[key]
        
        self.allPts = np.zeros((maxIters+2,self.nPts,self.nDims))
        self.allVelocities = np.zeros((maxIters+1,self.nPts,self.nDims))
        self.allForces = np.zeros((maxIters+1,self.nPts,self.nDims))

        self.allPts[0] = self.initialPoints
        self.allForces[0] = self.nebObj.compute_force(self.allPts[0])
        
        if useLocal:
            tStepArr = np.zeros((maxIters+1,self.nPts))
            tStepArr[:,:] = tStep
            alphaArr = np.zeros((maxIters+1,self.nPts))
            stepsSinceReset = np.zeros((self.nPts))
        else:
            stepsSinceReset = 0
            tStepArr = np.zeros(maxIters+1)
            alphaArr = np.zeros(maxIters+1)
        
        tStepArr[0] = tStep
        alphaArr[0] = fireParams["aStart"]
        
        endsWithoutError = True
        
        t0 = time.time()
        try:
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
                            
                if earlyStop:
                    stopBool = self._check_early_stop(step,earlyStopParams)
                    if stopBool:
                        self.allPts = self.allPts[:step+2]
                        self.allVelocities = self.allVelocities[:step]
                        self.allForces = self.allForces[:step]
                        
                        tStepArr = tStepArr[:step]
                        alphaArr = alphaArr[:step]
                        break
                    
                if earlyAbort:
                    stopBool = self._check_early_abort(step,earlyAbortParams)
                    if stopBool:
                        endsWithoutError = False
                        break
                    
            #Final iteration
            if useLocal:
                tStepFinal = tStepArr[-1].reshape((-1,1))
                shift = tStepFinal*self.allVelocities[-1] + \
                    0.5*self.allForces[-1]*tStepFinal**2
    
                for ptIter in range(self.nPts):
                    for dimIter in range(self.nDims):
                        if(abs(shift[ptIter,dimIter])>fireParams["maxmove"][dimIter]):
                            shift[ptIter] = shift[ptIter] * \
                                fireParams["maxmove"][dimIter]/abs(shift[ptIter,dimIter])
    
                self.allPts[-1] = self.allPts[-2] + shift
            else:
                self.allPts[-1] = self.allPts[-2] + tStepArr[-1]*self.allVelocities[-1] + \
                    0.5*self.allForces[-1]*tStepArr[-1]**2
        finally:
            t1 = time.time()
            self.nebObj.logger.flush()
            self.nebObj.logger.write_fire_params(tStepArr,alphaArr,stepsSinceReset,fireParams)
            self.nebObj.logger.write_runtime(t1-t0)
            if earlyStop:
                self.nebObj.logger.write_early_stop_params(earlyStopParams)
        
        return tStepArr, alphaArr, stepsSinceReset, endsWithoutError
    
    def _local_fire_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        tStepPrev = tStepArr[step-1].reshape((-1,1)) #For multiplication below
        
        shift = tStepPrev*self.allVelocities[step-1] + \
                0.5*self.allForces[step-1]*tStepPrev**2

        for ptIter in range(self.nPts):
            for dimIter in range(self.nDims):
                if(abs(shift[ptIter,dimIter])>fireParams["maxmove"][dimIter]):
                    shift[ptIter] = shift[ptIter] * \
                        fireParams["maxmove"][dimIter]/abs(shift[ptIter,dimIter])

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
        """
        Implements the FIRE algorithm from doi.org/10.1016/j.commatsci.2020.109584
        (algorithm 1) using a mixed semi-implicit Euler update (algorithm 4). Note
        that there is a typo in the mixing: line 2 should involve v(t + \Delta t) on
        the right-hand side, rather than v(t). Line 10 of algorithm 1 is ignored;
        performance appears to be way better (or even correct at all) when mixing
        occurs in the integration step

        Parameters
        ----------
        step : TYPE
            DESCRIPTION.
        tStepArr : TYPE
            DESCRIPTION.
        alphaArr : TYPE
            DESCRIPTION.
        stepsSinceReset : TYPE
            DESCRIPTION.
        fireParams : TYPE
            DESCRIPTION.

        Returns
        -------
        tStepArr : TYPE
            DESCRIPTION.
        alphaArr : TYPE
            DESCRIPTION.
        stepsSinceReset : TYPE
            DESCRIPTION.

        """
        vdotf = 0.
        for ptIter in range(self.nPts):
            vdotf += np.dot(self.allVelocities[step-1,ptIter],self.allForces[step-1,ptIter])
        
        if vdotf > 0:
            stepsSinceReset += 1
            
            if stepsSinceReset > fireParams["nAccel"]:
                tStepArr[step] = min(tStepArr[step-1]*fireParams["fInc"],fireParams["dtMax"])
                alphaArr[step] = alphaArr[step-1]**fireParams["fAlpha"]
            else:
                tStepArr[step] = tStepArr[step-1]
                alphaArr[step] = alphaArr[step-1]            
        else:
            stepsSinceReset = 0
            self.allVelocities[step-1,ptIter] = np.zeros(self.nDims)
            tStepArr[step] = max(tStepArr[step-1]*fireParams["fDecel"],fireParams["dtMin"])
            alphaArr[step] = fireParams["aStart"]
            
        #Semi-implicit Euler integration
        self.allVelocities[step] = self.allVelocities[step-1] + tStepArr[step]*self.allForces[step-1]
        
        vdotv = 0.
        fdotf = 0.
        for ptIter in range(self.nPts):
            vdotv += np.linalg.norm(self.allVelocities[step,ptIter])
            fdotf += np.linalg.norm(self.allForces[step-1,ptIter])
        
        if fdotf > 10**(-16):
            scale = vdotv/fdotf
        else:
            scale = 0.
        
        self.allVelocities[step] = (1-alphaArr[step]) * self.allVelocities[step]\
            + alphaArr[step] * scale * self.allForces[step-1]
        
        shift = tStepArr[step]*self.allVelocities[step]
        
        for ptIter in range(self.nPts):
            for dimIter in range(self.nDims):
                if(abs(shift[ptIter,dimIter])>fireParams["maxmove"][dimIter]):
                    shift[ptIter] = shift[ptIter] * \
                        fireParams["maxmove"][dimIter]/abs(shift[ptIter,dimIter])
        
        self.allPts[step] = self.allPts[step-1] + shift
        self.allForces[step] = self.nebObj.compute_force(self.allPts[step])
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def fire2(self,tStep,maxIters,fireParams={},useLocal=False,earlyStop=False,
              earlyStopParams={}):
        """
        Wrapper for fast inertial relaxation engine 2.
        FIRE step taken from http://dx.doi.org/10.1103/PhysRevLett.97.170201
        
        Velocity update taken from 
        https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet

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
            {"dtMax":10.,"dtMin":0.02,"nAccel":10,"fInc":1.1,"fAlpha":0.99,\
             "fDecel":0.5,"aStart":0.1,"maxmove":np.full(self.nDims,1.0),\
             "minDecelIter":20}
            
        for key in fireParams.keys():
            if key not in defaultFireParams.keys():
                raise ValueError("Key "+key+" in fireParams not allowed")
                
        for key in defaultFireParams.keys():
            if key not in fireParams.keys():
                fireParams[key] = defaultFireParams[key]
                
        defaultStopParams = {"startCheckIter":300,"nStabIters":50,"checkFreq":10,\
                             "stabPerc":0.002}
        
        for key in earlyStopParams.keys():
            if key not in defaultStopParams.keys():
                raise ValueError("Key "+key+" in earlyStopParams not allowed")
                
        for key in defaultStopParams.keys():
            if key not in earlyStopParams.keys():
                earlyStopParams[key] = defaultStopParams[key]
                
        self.allPts = np.zeros((maxIters+2,self.nPts,self.nDims))
        self.allVelocities = np.zeros((maxIters+1,self.nPts,self.nDims))
        self.allForces = np.zeros((maxIters+1,self.nPts,self.nDims))

        self.allPts[0] = self.initialPoints
        self.allForces[0] = self.nebObj.compute_force(self.allPts[0])
        
        if useLocal:
            tStepArr = tStep*np.ones((maxIters+1,self.nPts))
            alphaArr = np.zeros((maxIters+1,self.nPts))
            stepsSinceReset = np.zeros((self.nPts))
        else:
            stepsSinceReset = 0
            tStepArr = np.zeros(maxIters+1)
            alphaArr = np.zeros(maxIters+1)
        
        tStepArr[0] = tStep
        alphaArr[0] = fireParams["aStart"]
        
        t0 = time.time()
        try:
            for step in range(1,maxIters+1):
                if useLocal:
                    tStepArr,alphaArr,stepsSinceReset = \
                        self._local_fire2_iter(step,tStepArr,alphaArr,stepsSinceReset,\
                                               fireParams)
                else:
                    tStepArr,alphaArr,stepsSinceReset = \
                        self._global_fire2_iter(step,tStepArr,alphaArr,stepsSinceReset,\
                                                fireParams)
                            
                if earlyStop:
                    stopBool = self._check_early_stop(step,earlyStopParams)
                    if stopBool:
                        self.allPts = self.allPts[:step+2]
                        self.allVelocities = self.allVelocities[:step]
                        self.allForces = self.allForces[:step]
                        
                        tStepArr = tStepArr[:step]
                        alphaArr = alphaArr[:step]
                        break
            
            if useLocal:
                tStepFinal = tStepArr[-1].reshape((-1,1))
                shift = tStepFinal*self.allVelocities[-1] + \
                    0.5*self.allForces[-1]*tStepFinal**2
    
                for ptIter in range(self.nPts):
                    for dimIter in range(self.nDims):
                        if(abs(shift[ptIter,dimIter])>fireParams["maxmove"][dimIter]):
                            shift[ptIter] = shift[ptIter] * \
                                fireParams["maxmove"][dimIter]/abs(shift[ptIter,dimIter])
    
                self.allPts[-1] = self.allPts[-2] + shift
            else:
                self.allPts[-1] = self.allPts[-2] + tStepArr[-1]*self.allVelocities[-1] + \
                    0.5*self.allForces[-1]*tStepArr[-1]**2
        finally:
            t1 = time.time()
            self.nebObj.logger.flush()
            self.nebObj.logger.write_fire_params(tStepArr,alphaArr,stepsSinceReset,fireParams)
            self.nebObj.logger.write_runtime(t1-t0)
            if earlyStop:
                self.nebObj.logger.write_early_stop_params(earlyStopParams)
            
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _local_fire2_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        warnings.warn("Local FIRE2 currently calls local FIRE update")
        return self._local_fire_iter(step,tStepArr,alphaArr,stepsSinceReset,fireParams)
    
    def _global_fire2_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        """
        Implements the FIRE 2 algorithm from doi.org/10.1016/j.commatsci.2020.109584
        (algorithm 2) using a mixed semi-implicit Euler update (algorithm 4). Note
        that there is a typo in the mixing: line 2 should involve v(t + \Delta t) on
        the right-hand side, rather than v(t)

        Parameters
        ----------
        step : TYPE
            DESCRIPTION.
        tStepArr : TYPE
            DESCRIPTION.
        alphaArr : TYPE
            DESCRIPTION.
        stepsSinceReset : TYPE
            DESCRIPTION.
        fireParams : TYPE
            DESCRIPTION.

        Returns
        -------
        tStepArr : TYPE
            DESCRIPTION.
        alphaArr : TYPE
            DESCRIPTION.
        stepsSinceReset : TYPE
            DESCRIPTION.

        """
        vdotf = 0.0
        for ptIter in range(self.nPts):
            vdotf += np.dot(self.allVelocities[step-1,ptIter],self.allForces[step-1,ptIter])

        if vdotf > 0.0:
            stepsSinceReset += 1
            if stepsSinceReset > fireParams["nAccel"]:
                tStepArr[step] = \
                    min(tStepArr[step-1]*fireParams["fInc"],fireParams["dtMax"])
                alphaArr[step] = alphaArr[step-1]*fireParams["fAlpha"]
            else:
                tStepArr[step] = tStepArr[step-1]
                alphaArr[step] = alphaArr[step-1]
        else:
            alphaArr[step] = fireParams["aStart"]
            if(step > fireParams["minDecelIter"]):
                tStepArr[step] = \
                    max(tStepArr[step-1]*fireParams["fDecel"],fireParams["dtMin"])
            else:
                tStepArr[step] = tStepArr[step-1]
            self.allPts[step-1] = self.allPts[step-1] - 0.5*tStepArr[step]*self.allVelocities[step-1,:,:]
            self.allVelocities[step-1] = 0.0
            
        #Semi-implicit Euler integration
        self.allVelocities[step] = self.allVelocities[step-1] + tStepArr[step] * self.allForces[step-1]
        
        #For mixing
        vdotv = 0.0
        fdotf = 0.0
        for ptIter in range(self.nPts):
            vdotv += np.dot(self.allVelocities[step,ptIter],self.allVelocities[step,ptIter])
            fdotf += np.dot(self.allForces[step-1,ptIter],self.allForces[step-1,ptIter])
        #Only vanishes if net force on all particles is zero, in which case this is zeroed out
        #later anyways. Handled here to prevent nuisance exception-throwing
        if fdotf > 10**(-16):
            scale = np.sqrt(vdotv/fdotf)
        else:
            scale = 0.0
        
        self.allVelocities[step] = (1-alphaArr[step])*self.allVelocities[step] + \
            alphaArr[step] * scale * self.allForces[step-1]
        
        shift = tStepArr[step]*self.allVelocities[step]
        
        for ptIter in range(self.nPts):
            for dimIter in range(self.nDims):
                if(abs(shift[ptIter,dimIter])>fireParams["maxmove"][dimIter]):
                    shift[ptIter] = shift[ptIter] * \
                        fireParams["maxmove"][dimIter]/abs(shift[ptIter,dimIter])
        
        self.allPts[step] = self.allPts[step-1] + shift
        self.allForces[step] = self.nebObj.compute_force(self.allPts[step])
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _check_early_stop(self,currentIter,stopParams):
        """
        Computes standard deviation in location of every image over the previous
        stopParams["nStabIters"] iterations. Returns True if std is less than
        stopParams["stabPerc"] for every image.

        Parameters
        ----------
        currentIter : TYPE
            DESCRIPTION.
        stopParams : TYPE
            DESCRIPTION.

        Returns
        -------
        ret : TYPE
            DESCRIPTION.

        """
        ret = False
        
        startCheckIter = stopParams["startCheckIter"]
        stabPerc = stopParams["stabPerc"]
        nStabIters = stopParams["nStabIters"]
        checkFreq = stopParams["checkFreq"]
        
        if (currentIter >= startCheckIter) and (currentIter % checkFreq == 0):
            std = np.std(self.allPts[currentIter-nStabIters:currentIter],axis=0)
            if np.all(std <= stabPerc):
                ret = True
        
        return ret
    
    def _check_early_abort(self,currentIter,breakParams):
        
        ret = False
        
        startCheckIter = breakParams["startCheckIter"]
        allowedVariance = breakParams["variance"]
        nStabIters = breakParams["nStabIters"]
        checkFreq = breakParams["checkFreq"]
        
        if (currentIter >= startCheckIter) and (currentIter % checkFreq == 0):
            std = np.std(self.allPts[currentIter-nStabIters:currentIter],axis=0)
            if np.any(std >= allowedVariance):
                ret = True
        return ret
    
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
    
class EulerLagrangeVerifier:
    """
    :Maintainer: Daniel
    """
    def __init__(self,path,eneg_func,mass_func=None,grad_approx=midpoint_grad):
        """
        

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        eneg_func : TYPE
            DESCRIPTION.
        mass_func : TYPE, optional
            DESCRIPTION. The default is None.
        grad_approx : Function, optional
            For computing gradients of the potential and inertia tensor. 
            The default is midpoint_grad.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(path,np.ndarray):
            self.interpPath = InterpolatedPath(path)
        elif callable(interpPath):
            self.interpPath = path
        else:
            raise TypeError("path is neither an ndarray, nor callable")
        
        self.nPts, self.nDims = self.interpPath.path.shape
        
        self.eneg_func = eneg_func
        self.mass_func = mass_func
        self.grad_approx = grad_approx
    
    def _compare_lagrangian_id_inertia(self,x,dx,ddx):
        lhs = np.zeros((self.nPts,self.nDims))
        rhs = np.zeros((self.nPts,self.nDims))
        
        for ptIter in range(1,self.nPts-1):
            lhs[ptIter] = 2*self.eneg_func(x[ptIter])*ddx[ptIter]
            enegGrad = self.grad_approx(self.eneg_func,x[ptIter])
            
            term1 = enegGrad * np.dot(dx[ptIter],dx[ptIter])
            term2 = -dx[ptIter] * np.dot(enegGrad,dx[ptIter])
            
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
    
    def compare_lagrangian(self,nPts):
        t = np.linspace(0,1,nPts)
        x = self.interpPath(t)
        
        dx = np.zeros((self.nPts,self.nDims))
        for ptIter in range(1,self.nPts-1):
            tPlus = t[ptIter] + fdTol/2
            tMinus = t[ptIter] - fdTol/2
            dx[ptIter] = (self.interpPath(tPlus) - self.interpPath(tMinus))/(2*fdTol)
            
        ddx = np.zeros((self.nPts,self.nDims))
        for ptIter in range(1,self.nPts-1):
            tPlus = t[ptIter] + fdTol
            tMinus = t[ptIter] - fdTol
            ddx[ptIter] = (self.interpPath(tPlus) - 2*x[ptIter] + self.interpPath(tMinus))/fdTol
        
        if self.mass_func is None:
            elDiff = self._compare_lagrangian_id_inertia(x,dx,ddx)
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
                 trimVals=[10**(-4),None],logLevel=1,fName=None):
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
                                 "; required shape is "+str(inertArrRequiredShape))
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
        
        self.djkLogger = DijkstraLogger(self,logLevel=logLevel,fName=fName)
    
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
            #Removing visited indices
            neighborInds = [tuple(n) for n in neighborInds if not tentativeDistance.mask[tuple(n)]]
            
            if self.djkLogger.logLevel == 2:
                print(50*"*")
                print("Current inds: ",currentInds)
                print("Current point: ",np.array([c[currentInds] for c in self.coordMeshTuple]))
                print("Neighbor inds:\n",neighborInds)
                if self.potArr.size <= 50: #Any larger is too big to be useful
                    with np.printoptions(precision=2):
                        print("Updated data:\n",tentativeDistance.data)
                        print("Mask:\n",tentativeDistance.mask)
            
            #For feeding into self.target_func
            coords = np.zeros((2,self.nDims))
            coords[0] = np.array([c[currentInds] for c in self.coordMeshTuple])
            
            enegs = np.zeros(2)
            enegs[0] = self.potArr[currentInds]
            
            masses = np.zeros((2,)+2*(self.nDims,))
            masses[0] = self.inertArr[currentInds]
            
            for (neighIter, n) in enumerate(neighborInds):
                coords[1] = [c[n] for c in self.coordMeshTuple]
                enegs[1] = self.potArr[n]
                masses[1] = self.inertArr[n]
                
                #self.target_func returns the action (distance), plus energies and masses
                distThroughCurrent = tentativeDistance[currentInds] + \
                    self.target_func(coords,enegs,masses)[0]
                    
                if self.djkLogger.logLevel == 2:
                    print("Neighbor inds: ",n)
                    print("Neighbor point: ",coords[1])
                    print("Current distance: ",tentativeDistance[n])
                    print("Dist through current: ",distThroughCurrent)
                    print("Updating neighbor: ",distThroughCurrent<tentativeDistance[n])
                
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

class DynamicProgramming:
    def __init__(self,initialPoint,coordMeshTuple,potArr,inertArr=None,allowedMask=None,\
                 target_func=TargetFunctions.action,allowedEndpoints=None,\
                 trimVals=[10**(-4),None],logLevel=1,fName=None,logFreq=50):
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
        if allowedMask is None:
            self.allowedMask = np.ones(expectedShape,dtype=bool)
        else:
            if allowedMask.shape == expectedShape:
                self.allowedMask = allowedMask
            else:
                dummyArr = np.swapaxes(allowedMask,0,1)
                if dummyArr.shape == expectedShape:
                    self.allowedMask = dummyArr
                else:
                    raise ValueError("allowedMask.shape is "+str(allowedMask.shape)+\
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
            #Simplifies things if I set this to the identity here
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
                
        if np.any(self.allowedEndpoints[:,0]<=self.initialPoint[0]):
            raise ValueError("All final endpoints must have 0'th coordinate greater"\
                             +" than the 0'th coordinate of the initial point")
        
        self.endpointIndices = [tuple(row) for row in self.endpointIndices]
        
        #Clip the potential to the min/max. Done after finding possible endpoints.
        self.trimVals = trimVals
        if self.trimVals != [None,None]:
            self.potArr = self.potArr.clip(self.trimVals[0],self.trimVals[1])
        else:
            warnings.warn("Not clipping self.potArr; may run into negative numbers in self.target_func")
        
        #Getting indices for self.initialPoint
        self.initialInds = np.zeros(self.nDims,dtype=int)
        for dimIter in range(self.nDims):
            #More nuisances with floating-point precision. Should maybe find
            #source of the issue, but this works for the basic test case.
            self.initialInds[dimIter] = \
                np.argwhere(np.isclose(self.uniqueCoords[dimIter],self.initialPoint[dimIter]))
        
        self.uniqueSliceInds = [np.arange(self.potArr.shape[0]),[]]
        for s in self.potArr.shape[2:]:
            self.uniqueSliceInds.append([np.arange(s)])
        
        #Index swapping like mentioned above. Now, self.initialInds[0] <= Ny,
        #and self.initialInds[1] <= Nx
        self.initialInds[[1,0]] = self.initialInds[[0,1]]
        self.initialInds = tuple(self.initialInds)
        
        self.logger = DPMLogger(self,logLevel=logLevel,fName=fName)
        self.logFreq = logFreq
    
    def _gen_slice_inds(self,constInd):
        sliceCopy = self.uniqueSliceInds.copy()
        sliceCopy[1] = [constInd]
        
        return list(itertools.product(*sliceCopy))
    
    def _select_prior_points(self,currentIdx,previousIndsArr,distArr):
        previousInds = self._gen_slice_inds(currentIdx-1)
        #Use scipy.ndimage.label to only select previous indices that are connected
        #to the current one. Imperfect - on vertical OTL, will choose from far
        #away points - but unclear if/when that happens. More sophisticated
        #method could look at connected current/previous slice; when they disagree,
        #set a range for how far away the index can be on the previous slice.
        #Won't be perfect. I still expect paths can follow the LAP, then jump
        #around at the end, but they will no longer pass over the region outside
        #the OTL, except for a bit near the OTL. "Connected" in this case means
        #the mask saying which indices are allowed is connected in these two slices.
        currentInds = self._gen_slice_inds(currentIdx)
        
        for idx in currentInds:
            coords = np.zeros((2,self.nDims))
            coords[1] = [c[idx] for c in self.coordMeshTuple]
            
            enegs = np.zeros((2,))
            enegs[1] = self.potArr[idx]
            if enegs[1] == np.inf:
                continue
            
            masses = np.zeros((2,self.nDims,self.nDims))
            masses[1] = self.inertArr[idx]
            
            for p in previousInds:
                coords[0] = [c[p] for c in self.coordMeshTuple]
                enegs[0] = self.potArr[p]
                if enegs[0] == np.inf:
                    continue
                masses[0] = self.inertArr[p]
                
                tentDist = distArr[p] + self.target_func(coords,enegs,masses)[0]
                if tentDist < distArr[idx]: #distArr is initialized to infinity
                    previousIndsArr[idx] = p
                    distArr[idx] = tentDist
        
        return previousIndsArr, distArr
    
    def __call__(self,searchRange=None,pathAsText=True):
        # if searchRange is None:
        #     uniqueSliceInds = [np.arange(self.potArr.shape[0]),[]]
        #     for s in self.potArr.shape[2:]:
        #         uniqueSliceInds.append([np.arange(s)])
        # elif 
        
        t0 = time.time()
        
        previousIndsArr = -1*np.ones(self.potArr.shape+(self.nDims,),dtype=int)
        previousIndsArr[self.initialInds] = self.initialInds
        previousIndsArr[:,self.initialInds[1]+1] = self.initialInds
        
        distArr = np.inf*np.ones(self.potArr.shape)
        distArr[self.initialInds] = 0
        
        #Main loop. Because distArr is initialized to np.inf except at the origin,
        #we don't have to initialize the first column separately.
        finalIdx = np.max(np.array(self.endpointIndices)[:,1])
        
        for q2Idx in range(self.initialInds[1]+1,finalIdx+1):
            previousIndsArr, distArr = \
                self._select_prior_points(q2Idx,previousIndsArr,distArr)
            if q2Idx % self.logFreq == 0:
                updateRange = (q2Idx-self.logFreq,q2Idx)
                self.logger.log(previousIndsArr,distArr,updateRange)
        
        updateRange = (finalIdx-self.logFreq,finalIdx) #Some overlap here but whatever
        self.logger.log(previousIndsArr,distArr,updateRange)
        
        #Getting paths given previousIndsArr
        minIndsDict = {}
        minPathDict = {}
        distsDict = {}
        for endInds in self.endpointIndices:
            key = tuple([c[endInds] for c in self.coordMeshTuple])
            
            distsDict[key] = distArr[endInds]
            
            path = []
            ind = endInds
            
            while ind != self.initialInds:
                if ind == self.nDims*(-1,):
                    break
                    #raise ValueError("Reached invalid index "+str(ind))
                path.append(ind)
                ind = tuple(previousIndsArr[ind])
                
            path.append(self.initialInds)
            path.reverse()
            
            minIndsDict[key] = path
            minPathDict[key] = np.array([[c[i] for c in self.coordMeshTuple] for\
                                         i in path])
        
        t1 = time.time()
        
        self.logger.finalize(minPathDict,minIndsDict,distsDict,t1-t0,\
                             pathAsText=pathAsText)
        
        return minIndsDict, minPathDict, distsDict
