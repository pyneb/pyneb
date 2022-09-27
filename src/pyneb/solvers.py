import numpy as np

import itertools

from scipy.integrate import solve_bvp

import h5py
import sys
import time
import warnings

from utilities import *
from fileio import *

class LeastActionPath:
    """
    Computes the net force on a band, when minimizing an action-type functional,
    of the form
    
    $$ S = \int_{s_0}^{s_1} \sqrt{2 M_{ij}\dot{x}_i\dot{x}_j E(x(s))} ds. $$
    
    Can be generalized to functionals of the form
    
    $$ S = \int_{s_0}^{s_1} f(s) ds $$
    
    by choosing target_func differently. A common example is minimizing
    
    $$ S = \int_{s_0}^{s_1} M_{ij}\dot{x}_i\dot{x}_j E(x(s)) ds, $$
    
    with no square root inside of the integral.
    
    Attributes
    ----------
    potential : function
        Evaluates the energy along the path
    nPts : int
        The number of images in the path (path.shape[0])
    nDims : int
        The number of coordinates (path.shape[1])
    mass : function
        Evaluates the metric tensor M_{ij}
    endpointSpringForce : bool or list of bools
        Whether to turn on the endpoint spring force. Can be controlled for
        both endpoints individually. The elements correspond to the first and 
        the last endpoint
    endpointHarmonicForce : bool or list of bools
        The same as endpointSpringForce, except for the harmonic force term. Disabling
        both forces keeps an endpoint fixed
    target_func : function
        The functional to be minimized
    target_func_grad : function
        The approximation of the gradient of target_func
    logger : instance of fileio.ForceLogger
        Handles storing data collected during run
    nebParams : dict
        Contains the spring and harmonic force strengths, and the energy
        the endpoints are constrained to. Maintained for compatibility with
        self.logger
    k : float
        The spring force parameter
    kappa : float
        The harmonic force parameter
    constraintEneg : float
        The energy the endpoints are constrained to
    
    Methods
    -------
    compute_force(points)
        Computes the net force at every point in points
    
    :Maintainer: Daniel
    """
    def __init__(self,potential,nPts,nDims,mass=None,endpointSpringForce=True,\
                 endpointHarmonicForce=True,target_func=TargetFunctions.action,\
                 target_func_grad=GradientApproximations().forward_action_grad,\
                 nebParams={},logLevel=1,loggerSettings={}):
        """
        Parameters
        ----------
        potential : function
            Evaluates the energy along the path. To be called as potential(path). 
            Is passed to "target_func".
        nPts : int
            The number of images in the path (path.shape[0])
        nDims : int
            The number of coordinates (path.shape[1])
        mass : function, optional
            Evaluates the metric tensor M_{ij} along the path. To be called as mass(path). 
            Is passed to "target_func".The default is None, in which case $M_{ij}$ 
            is treated as the identity matrix at all points
        endpointSpringForce : bool or list of bools, optional
            Whether to turn on the endpoint spring force. Can be controlled for
            both endpoints individually. If a list of bools, the elements
            correspond to the first and the last endpoint. If a single bool, is applied
            to both endpoints. The default is True.
        endpointHarmonicForce : bool or list of bools, optional
            The same as endpointSpringForce, except for the harmonic force term. Disabling
            both forces keeps an endpoint fixed. The default is True.
        target_func : function, optional
            The functional to be minimized. Should take as arguments
            (path, potential, mass). Should return (action, potentialAtPath, massesAtPath).
            The default is utilities.TargetFunctions.action
        target_func_grad : function, optional
            The approximation of the gradient of target_func. Should take as arguments 
                (path, potentialFunc, potentialOnPath, massFunc, massOnPath, target_func),
            where target_func is the action integral approximation. Should return 
            (gradOfAction, gradOfPes). The default is
            utilities.GradientApproximations().forward_action_grad
        nebParams : dict, optional
            Contains the spring force and the harmonic oscillator potential parameters,
            as well as the energy the endpoints are constrained to. The default is {}, 
            in which case the parameters are
                {"k":10,"kappa":20,"constraintEneg":0}
        logLevel : int, optional
            Controls how much information is tracked. Level 0 turns off logging.
            See fileio.ForceLogger, or a .lap file, for documentation on other
            tracked information
        loggerSettings : dict, optional
            See fileio.ForceLogger for documentation
            
        Raises
        ------
        ValueError
            If one of endpointSpringForce or endpointHarmonicForce is not
            supplied as a bool, or a list or tuple of bools

        """
        #TODO: consider not having NEB parameters as a dictionary
        defaultNebParams = {"k":10,"kappa":20,"constraintEneg":0}
        for key in defaultNebParams.keys():
            if key not in nebParams:
                nebParams[key] = defaultNebParams[key]
        
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
                             " for endpointHarmonicForce")
        
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
        Computes tangent vectors along the path
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
        Computes the spring force along the path. Equations taken from 
        https://doi.org/10.1063/1.5007180 eqns 20-22
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
        """
        Computes the net force along the path

        Parameters
        ----------
        points : np.ndarray
            The path to evaluate the force along. Of shape (self.nPts,self.nDims)

        Returns
        -------
        netForce : np.ndarray
            The force at each image on the path. Of shape (self.nPts,self.nDims)

        """
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
                
        negIntegGrad = -gradOfAction.copy()
        trueForce = -gradOfPes.copy()
        
        projection = np.array([np.dot(negIntegGrad[i],tangents[i]) \
                               for i in range(self.nPts)])

        parallelForce = np.array([projection[i]*tangents[i] for i in range(self.nPts)])

        
        perpForce = negIntegGrad - parallelForce
        springForce = self._spring_force(points,tangents)
        
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            netForce[i] = perpForce[i] + springForce[i]
        #TODO: add check if force is very small
        
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
                 target_func_grad=GradientApproximations().mep_default,nebParams={},\
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
        if not isinstance(endpointSpringForce,(tuple,list)):
            raise ValueError("Unknown value "+str(endpointSpringForce)+\
                             " for endpointSpringForce")
                
        if isinstance(endpointHarmonicForce,bool):
            endpointHarmonicForce = 2*(endpointHarmonicForce,)
        if not isinstance(endpointHarmonicForce,(tuple,list)):
            raise ValueError("Unknown value "+str(endpointHarmonicForce)+\
                             " for endpointHarmonicForce")
        
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
    Iterative algorithms for minimizing the force along a path, e.g. the least
    action path, or a minimum energy path.
    
    Attributes
    ----------
    nebObj : object
        A class instance with method compute_force. Typically an instance of
        LeastActionPath or MinimumEnergyPath
    initialPoints : np.ndarray
        The starting path to iterate from
    nPts : int
        The length of the path, initialPoints.shape[0]
    nDims : int
        The number of coordinates, initialPoints.shape[1]
    allPts : np.ndarray
        The points at every iteration. Of shape [:,nPts,nDims]
    allVelocities : np.ndarray
        The velocity at every iteration. Of shape [:,nPts,nDims]
    allForces : np.ndarray
        The net force at every iteration. Of shape [:,nPts,nDims]
    
    Methods
    -------
    velocity_verlet(tStep,maxIters,dampingParameter=0)
        Uses a velocity verlet algorithm, with optional damping parameter
    fire(tStep,maxIters,**kwargs)
        The Fast Inertial Relaxation Algorithm, with adaptive timestep. Timestep
        can be same for all points, or vary with the points
    fire2(tStep,maxIters,**kwargs)
        Similar to fire, but with different correction when overstepping in wrong
        direction
    
    :Maintainer: Daniel
    """
    def __init__(self,nebObj,initialPoints):
        """
        Parameters
        ----------
        nebObj : object
            A class instance with method compute_force. Typically an instance of
            LeastActionPath or MinimumEnergyPath
        initialPoints : np.ndarray
            The starting path to iterate from

        Raises
        ------
        AttributeError
            If nebObj has no attribute compute_force
        ValueError
            If nebObj.nPts != initialPoints.shape[0]
        """
        
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
        
    @np.errstate(all="raise")
    def velocity_verlet(self,tStep,maxIters,dampingParameter=0):
        """
        The velocity Verlet algorithm, taken from Algorithm 6 of 
        https://doi.org/10.1021/acs.jctc.7b00360
        Has an optional damping force.
        
        Parameters
        ----------
        tStep : float
            The timestep in the algorithm. Suggested value is < 1
        maxIters : int
            The maximum number of iterations
        dampingParameter : float, optional
            Controls how much the algorithm slows down with time, by stepping
            backwards according to the velocity at the current time step with
            strength dampingParameter. The default is 0

        Returns
        -------
        endsWithoutError : bool
            If the algorithm completes without error. For cases such as LeastActionPath,
            data is logged to the output file regardless of this status
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
            
            if sys.exc_info()[0] is None:
                endsWithoutError = True
            else:
                endsWithoutError = False
            
            t1 = time.time()
            self.nebObj.logger.flush()
            self.nebObj.logger.write_runtime(t1-t0)
            self.nebObj.logger.write_run_params("verlet_params",
                                                {"tStep":tStep,"maxIters":maxIters,
                                                 "dampingParameter":dampingParameter})
            
            return endsWithoutError
    
    @np.errstate(all="raise")
    def fire(self,tStep,maxIters,fireParams={},useLocal=True,earlyStop=True,
             earlyStopParams={},earlyAbort=False,earlyAbortParams={}):
        """
        Wrapper for fast inertial relaxation engine. FIRE step taken from 
        http://dx.doi.org/10.1103/PhysRevLett.97.170201 Velocity update taken 
        from https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        FIRE has an adaptive timestep, that increases when travelling along the
        desired trajectory towards the minimum-desired path. When the path 
        goes too far in that direction, progress is reset to the previous iteration,
        and the velocity is zeroed out

        Parameters
        ----------
        tStep : float
            The timestep in the algorithm. Suggested value is < 1
        maxIters : int
            The maximum number of iterations
        fireParams : dict, optional
            The FIRE params. The default is {}, in which case the parameters
            are taken from defaultFireParams
        useLocal : bool, optional
            Whether to use a different timestep for every point on the path. 
            The default is True
        earlyStop : bool, optional
            Whether to stop early if convergence is reached in less than maxIters
            steps. The default is True
        earlyStopParams : dict, optional
            Parameters controlling early stopping. The default is {}, in which
            case the parameters are taken from defaultStopParams
        earlyAbort : bool, optional
            Whether to stop early if the path starts varying wildly away from
            convergence. The default is False
        earlyAbortParams : dict, optional
            Parameters controlling early aborting (i.e. stopping due to presumed error).
            The default is {}, in which case the parameters are taken from
            defaultAbortParams

        Raises
        ------
        ValueError
            If a parameter supplied to fireParams, earlyStopParams, or earlyAbortParams
            is not an expected parameter. Expected parameters are given in
            the dict of default parameters

        Returns
        -------
        tStepArr : np.ndarray
            The time step, as a function of iteration number. Contains the time
            step for each point separately, if useLocal is True
        alphaArr : np.ndarray
            The FIRE parameter alpha, as a function of iteration number. Contains 
            alpha for each point separately, if useLocal is True
        stepsSinceReset : int or np.ndarray
            How long the path has been travelling in the right direction. Is
            an array if useLocal is True, in which case each point is tracked
            separately
        endsWithoutError : bool
            If the algorithm completes without error. For cases such as LeastActionPath,
            data is logged to the output file regardless of this status
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
            
            self.nebObj.logger.write_run_params("fire_params",fireParams.update("useLocal",useLocal))
            if earlyStop:
                self.nebObj.logger.write_run_params("early_stop_params",earlyStopParams)
            if earlyAbort:
                self.nebObj.logger.write_run_params("early_abort_params",earlyAbortParams)
        
            return tStepArr, alphaArr, stepsSinceReset, endsWithoutError
    
    def _local_fire_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        """
        A single iteration for fire, using a different timestep for each point
        """
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
        A single iteration for fire, using the same timestep for each point
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
    
    @np.errstate(all="raise")
    def fire2(self,tStep,maxIters,fireParams={},useLocal=False,earlyStop=False,
              earlyStopParams={}):
        """
        Wrapper for fast inertial relaxation engine 2. FIRE step taken from 
        http://dx.doi.org/10.1103/PhysRevLett.97.170201 Velocity update taken 
        from https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        FIRE 2 is an adaptive algorithm, like FIRE, but when the path goes
        too far in one direction, only half of a step back is taken, and the
        velocity is zeroed out

        Parameters
        ----------
        tStep : float
            The timestep in the algorithm. Suggested value is < 1
        maxIters : int
            The maximum number of iterations
        fireParams : dict, optional
            The FIRE params. The default is {}, in which case the parameters
            are taken from defaultFireParams
        useLocal : bool, optional
            Whether to use a different timestep for every point on the path. 
            The default is False
        earlyStop : bool, optional
            Whether to stop early if convergence is reached in less than maxIters
            steps. The default is False
        earlyStopParams : dict, optional
            Parameters controlling early stopping. The default is {}, in which
            case the parameters are taken from defaultStopParams

        Raises
        ------
        ValueError
            If a parameter supplied to fireParams or earlyStopParams is not an 
            expected parameter. Expected parameters are given in the dict of 
            default parameters

        Returns
        -------
        tStepArr : np.ndarray
            The time step, as a function of iteration number. Contains the time
            step for each point separately, if useLocal is True
        alphaArr : np.ndarray
            The FIRE parameter alpha, as a function of iteration number. Contains 
            alpha for each point separately, if useLocal is True
        stepsSinceReset : int or np.ndarray
            How long the path has been travelling in the right direction. Is
            an array if useLocal is True, in which case each point is tracked
            separately
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
            
            self.nebObj.logger.write_run_params("fire_params",fireParams.update("useLocal",useLocal))
            if earlyStop:
                self.nebObj.logger.write_run_params("early_stop_params",earlyStopParams)
            # if earlyAbort:
            #     self.nebObj.logger.write_run_params("early_abort_params",earlyAbortParams)
            
            return tStepArr, alphaArr, stepsSinceReset
    
    def _local_fire2_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        """
        A holding cell for a single fire2 iteration with a different timestep for each
        point on the path. Currently calls the local fire update
        """
        warnings.warn("Local FIRE2 currently calls local FIRE update")
        return self._local_fire_iter(step,tStepArr,alphaArr,stepsSinceReset,fireParams)
    
    def _global_fire2_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        """
        A single iteration for fire2, using the same timestep for each point
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
        stopParams["stabPerc"] for every image, in which case the algorithm
        is said to have converged, and can stop
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
        """
        Computes standard deviation in location of every image over the previous
        breakParams["nStabIters"] iterations. Returns True if std is greater than
        breakParams["variance"] for any image, in which case the algorithm
        is said to be diverging, and should stop rather than proceed in a wrong
        direction
        """
        
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

class Dijkstra:
    """
    Implements Dijkstra's algorithm for finding the shortest path on a rectangular
    grid. Paths can connect gridpoints that are 0, +/- 1 away from each other in
    any direction
    
    Attributes
    ----------
    nDims : int
        The number of dimensions of the grid
    uniqueCoords : list of np.ndarray
        The unique coordinates of the grid
    coordMeshTuple : tuple of np.ndarray
        All of the coordinates, arranged as the output of np.meshgrid(*uniqueCoords)
    potArr : np.ndarray
        The energy at every point on the grid
    inertArr : np.ndarray
        The inertia at every point on the grid
    initialPoint : np.ndarray
        The starting coordinates
    initialInds : np.ndarray
        The indices of the starting coordinates
    allowedEndpoints : np.ndarray
        The possible ending coordinates
    endpointIndices : np.ndarray
        The indices of the possible ending coordinates
    target_func : function
        The functional to minimize
    djkLogger : instance of fileio.DijkstraLogger
        Handles storing data collected during run
    
    Methods
    -------
    __call__(returnAll=False)
        Runs main algorithm
    minimum_endpoint(distanceDict)
        Once Dijkstra's algorithm is complete, selects the endpoint with minimum
        target_func out of all possible endpoints
    
    :Maintainer: Daniel
    """
    def __init__(self,initialPoint,coordMeshTuple,potArr,inertArr=None,\
                 target_func=TargetFunctions.action,allowedEndpoints=None,\
                 trimVals=[10**(-4),None],logLevel=1,fName=None):
        """
        Parameters
        ----------
        initialPoint : np.ndarray
            The starting coordinates
        coordMeshTuple : tuple of np.ndarray
            All of the coordinates, arranged as the output of np.meshgrid
        potArr : np.ndarray
            The energy at every point on the grid
        inertArr : np.ndarray, optional
            The inertia at every point on the grid. The default is None, in
            which case the inertia is taken as the identity at every point
            on the grid
        target_func : function, optional
            The functional to minimize. The default is utilities.TargetFunctions.action
        allowedEndpoints : np.ndarray, optional
            Possible endpoints for the path. The default is None, in which case
            the endpoints are found using utilities.SurfaceUtils.find_endpoints_on_grid
        trimVals : list, optional
            The energy values to trim potArr by. The default is [10**(-4),None],
            so the maximum value is left unchanged

        Raises
        ------
        ValueError
            Throws if potArr, inertArr, or allowedEndpoints are an incorrect shape
        
        Notes
        -----
        Some indexing is done to deal with the default shape of np.meshgrid.
        For D dimensions, the output is of shape (N2,N1,N3,...,ND), while the
        way indices are generated expects a shape of (N1,...,ND). So, we swap
        the first two indices by hand. See 
        https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        
        Note that indexing for Dijkstra internal functions are done in the
        order (N2,N1,N3,...), for simplicity. The indexing that is returned
        by self.__call__ is kept in this order by default.
        
        Note that the value of the array at a certain index is the same
        regardless of the sort order of the indices, provided that the index
        order matches that used when creating np.meshgrid
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
                    raise ValueError("coordMeshTuple has wrong dimensions")
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
        if trimVals != [None,None]:
            self.potArr = self.potArr.clip(trimVals[0],trimVals[1])
        
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

        TODO: allow for non-grid PES. Simplest to have a list of indices that are not allowed,
        and set the mask to "visited" for those points, so that they are never
        even considered.
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
        """
        Given the dictionary describing where each node comes from, constructs
        the paths from the starting point to each possible endpoint
        """
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
        Runs Dijkstra's algorithm

        Parameters
        ----------
        returnAll : bool, optional
            Whether to return the paths for every endpoint. Default is False,
            in which case only the shortest path is returned

        Returns
        -------
        pathIndsDictRet (pathInds) : dict (np.ndarray)
            If returnAll, is dict of the form {endpoint:indices of path to endpoint}.
            If not returnAll, is indices of path to endpoint for shortest path
        pathArrDict (pathArr) : dict (np.ndarray)
            Same as pathIndsDictRet, but with the path itself, rather than the
            indices through the array
        distanceDict (dist) : dict (float)
            If returnAll, is dict of the form {endpoint:distance to endpoint}.
            If not returnAll, is the distance along the shortest path

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
        """
        Selects the endpoint with the minimal distance

        Parameters
        ----------
        distanceDict : dict
            For each visited node, contains the source for that node

        Returns
        -------
        endptOut : tuple
            The endpoint with the minimal distance

        """
        minDist = np.inf
        for (endpt, dist) in distanceDict.items():
            if dist < minDist:
                endptOut = endpt
                minDist = dist
        self.djkLogger.log((endptOut,),("endptOut",))
        return endptOut

class DynamicProgramming:
    """
    Implements the dynamic programming algorithm for finding the shortest path 
    on a rectangular grid. Paths can only move from left to right in the first
    coordinate
    
    Attributes
    ----------
    nDims : int
        The number of dimensions of the grid
    uniqueCoords : list of np.ndarray
        The unique coordinates of the grid
    coordMeshTuple : tuple of np.ndarray
        All of the coordinates, arranged as the output of np.meshgrid(*uniqueCoords)
    potArr : np.ndarray
        The energy at every point on the grid
    inertArr : np.ndarray
        The inertia at every point on the grid
    initialPoint : np.ndarray
        The starting coordinates
    initialInds : np.ndarray
        The indices of the starting coordinates
    allowedEndpoints : np.ndarray
        The possible ending coordinates
    endpointIndices : np.ndarray
        The indices of the possible ending coordinates
    target_func : function
        The functional to minimize
    uniqueSliceInds : list
        Contains indices labelling the previous slice when moving from left to
        right, with an empty list for the second coordinate
    logger : instance of fileio.DPMLogger
        Handles storing data collected during run
    logFreq : int
        How frequently logger writes to file
    
    Methods
    -------
    __call__(pathAsText=False)
        Runs main algorithm
    
    :Maintainer: Daniel
    """
    def __init__(self,initialPoint,coordMeshTuple,potArr,inertArr=None,\
                 target_func=TargetFunctions.action,allowedEndpoints=None,\
                 trimVals=[10**(-4),None],logLevel=1,fName=None,logFreq=50):
        """
        Parameters
        ----------
        initialPoint : np.ndarray
            The starting coordinates
        coordMeshTuple : tuple of np.ndarray
            All of the coordinates, arranged as the output of np.meshgrid
        potArr : np.ndarray
            The energy at every point on the grid
        inertArr : np.ndarray, optional
            The inertia at every point on the grid. The default is None, in
            which case the inertia is taken as the identity at every point
            on the grid
        target_func : function, optional
            The functional to minimize. The default is utilities.TargetFunctions.action
        allowedEndpoints : np.ndarray, optional
            Possible endpoints for the path. The default is None, in which case
            the endpoints are found using utilities.SurfaceUtils.find_endpoints_on_grid
        trimVals : list, optional
            The energy values to trim potArr by. The default is [10**(-4),None],
            so the maximum value is left unchanged
        logLevel : int, optional
            Controls how much information is tracked. Level 0 turns off logging.
            See fileio.DPMLogger for documentation on other tracked information
        fName : str, optional
            The log file name. The default is None, in which case the name is
            chosen to be the start time of the run
        logFreq : int, optional
            When moving from left to right on a grid, is how many slices
            are passed before writing to log file. The default is 50

        Raises
        ------
        ValueError
            Throws if potArr, inertArr, or allowedEndpoints are an incorrect shape
        ValueError
            Throws if the start point is to the right of any of allowedEndpoints
        
        Notes
        -----
        Some indexing is done to deal with the default shape of np.meshgrid.
        For D dimensions, the output is of shape (N2,N1,N3,...,ND), while the
        way indices are generated expects a shape of (N1,...,ND). So, we swap
        the first two indices by hand. See 
        https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        
        Note that indexing for DynamicProgramming internal functions are done in the
        order (N2,N1,N3,...), for simplicity. The indexing that is returned
        by self.__call__ is kept in this order by default.
        
        Note that the value of the array at a certain index is the same
        regardless of the sort order of the indices, provided that the index
        order matches that used when creating np.meshgrid
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
        if trimVals != [None,None]:
            self.potArr = self.potArr.clip(trimVals[0],trimVals[1])
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
        """
        Returns all indices on a slice whose second index is constInd
        """
        sliceCopy = self.uniqueSliceInds.copy()
        sliceCopy[1] = [constInd]
        
        return list(itertools.product(*sliceCopy))
    
    def _select_prior_points(self,currentIdx,previousIndsArr,distArr):
        """
        For all points in a given slice, selects the point in the previous slice
        that connects to the point in the current slice
        """
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
    
    def __call__(self,pathAsText=False):
        """
        Runs the dynamic programming algorithm

        Parameters
        ----------
        pathAsText : bool, optional
            Whether to save the final path to a .txt file. The default is False

        Returns
        -------
        minIndsDict : dict
            Is dict of the form {endpoint:indices of path to endpoint}
        minPathDict : dict 
            Same as minIndsDict, but with the path itself, rather than the
            indices through the array
        distsDict : dict
            Is dict of the form {endpoint:distance to endpoint}

        """
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
