from fileio import *

import numpy as np
import numdifftools as nd
import sys
import matplotlib.pyplot as plt
import itertools

from scipy.interpolate import interpnd, RectBivariateSpline, splprep, splev
from scipy.ndimage import filters, morphology, map_coordinates #For minimum finding
from pathos import helpers
from pathos.multiprocessing import ProcessingPool as Pool
import warnings

global fdTol
fdTol = 10**(-8)

class TargetFunctions:
    """
    Class containing integral-type functionals that are commonly minimized.
    Generically of the form

    $$ S = \int_{s_0}^{s_1} f(s) ds. $$

    Functionals are discretized as

    $$ S \approx \sum_{i=1}^{nPoints} f(s_i) (s_i-s_{i-1}). $$

    Methods
    -------
    action(points,potential,masses)
        The standard action functional
    action_squared(points,potential,masses)
        The action functional, but without a square root
    mep_default(points,potential,masses)
        Wrapper to be used when finding the minimum energy path. See solvers.MinimumEnergyPath for
        documentation on this unique case

    :Maintainer: Daniel
    """
    @staticmethod
    def action(path,potential,masses=None):
        """
        The standard action functional

        $$ S = \int_{s_0}^{s_1} \sqrt{2 M_{ij}\dot{x}_i\dot{x}_j E(x(s))} ds $$

        Parameters
        ----------
        path : np.ndarray
            The path to evaluate the action along
        potential : np.ndarray or function
            The energy along path. If an array, is the energy values; if a function, evaluates the
            energy along the path
        masses : None, np.ndarray, or function, optional
            The collective inertia along the path. If an array, is the inertia values; if a function,
            evaluates the inertia along the path. If is None, the inertia is the identity at all
            points along the path. The default is None

        Raises
        ------
        ValueError
            If one of potential or masses is np.ndarray, and not the same length as path

        Returns
        -------
        actOut : float
            The action value
        potArr : np.ndarray
            The energy values for each point in path
        massArr : np.ndarray
            The collective inertia tensor for each point in path

        Notes
        -----
        Trims the potential, and the distance between points, to have a minimum value of zero, rather
        than throwing an error

        :Maintainer: Daniel
        """

        nPoints, nDims = path.shape

        if masses is None:
            massArr = np.full((nPoints,nDims,nDims),np.identity(nDims))
        elif not isinstance(masses,np.ndarray):
            massArr = masses(path)
        else:
            massArr = masses

        massDim = (nPoints, nDims, nDims)
        if massArr.shape != massDim:
            raise ValueError("Dimension of massArr is "+str(massArr.shape)+\
                             "; required shape is "+str(massDim)+". See action function.")

        if not isinstance(potential,np.ndarray):
            potArr = potential(path)
        else:
            potArr = potential

        potShape = (nPoints,)
        if potArr.shape != potShape:
            raise ValueError("Dimension of potArr is "+str(potArr.shape)+\
                             "; required shape is "+str(potShape)+". See action function.")

        #TODO: check if we actually want this. Maybe with a warning?
        #Changed from np.clip due to https://github.com/numpy/numpy/issues/14281
        potArr = np.core.umath.clip(potArr,0,potArr.max())
        # potArr = potArr.clip(0)
        
        coordDiff = np.diff(path,axis=0)
        dist = np.einsum("ij,ijk,ik->i",coordDiff,massArr[1:],coordDiff) #The M_{ab} dx^a dx^b bit
        dist = np.core.umath.clip(dist,0,dist.max())
        # dist = dist.clip(0)
        actOut = np.sum(np.sqrt(2*dist*potArr[1:]))
        
        return actOut, potArr, massArr

    @staticmethod
    def _term_in_action_sum(points,potential,masses=None):
        """
        One of the discrete terms in the functional computed in "action"
        """
        nDims = points.shape[1]
        if points.shape[0] != 2:
            raise ValueError("Expected exactly 2 points; received "+str(points.shape[0]))

        if masses is None:
            massArr = np.identity(nDims)
        elif not isinstance(masses,np.ndarray):
            massArr = masses(points[1]).reshape((nDims,nDims))
        else:
            massArr = masses

        massDim = (nDims,nDims)
        if massArr.shape != massDim:
            raise ValueError("Dimension of massArr is "+str(massArr.shape)+\
                             "; required shape is "+str(massDim))

        #There's lots of things that can be fed in, including odd things like
        #np.int objects. These aren't caught with a simple filter like "isinstance(potential,int)",
        #so I'm trying it this way for a bit
        try:
            potArr = potential(points[1])
        except TypeError:
            if not isinstance(potential,np.ndarray):
                potArr = np.array([potential])
            else:
                potArr = potential

        potShape = (1,)
        if potArr.shape != potShape:
            raise ValueError("Dimension of potArr is "+str(potArr.shape)+\
                             "; required shape is "+str(potShape))

        #TODO: check if we want this
        potArr = potArr.clip(0)

        #Actual calculation
        coordDiff = points[1] - points[0]
        #The M_{ab} dx^a dx^b bit
        dist = np.dot(coordDiff,np.dot(massArr,coordDiff))
        if dist<0:
                dist = 0
        actOut = np.sqrt(2*potArr[0]*dist)

        return actOut, potArr, massArr
    
    @staticmethod
    def action_squared(path,potential,masses=None):
        '''
        The functional

        $$ S = \int_{s_0}^{s_1} M_{ij}\dot{x}_i\dot{x}_j E(x(s)) ds $$

        Parameters
        ----------
        See :py:func:`action`

        Raises
        ------
        ValueError
            See :py:func:`action`

        Returns
        -------
        actOut : float
            The functional value along the path
        potArr : np.ndarray
            The energy values for each point in path
        massArr : np.ndarray
            The collective inertia tensor for each point in path


        :Maintainer: Eric
        '''

        nPoints, nDims = path.shape

        if masses is None:
            massArr = np.full((nPoints,nDims,nDims),np.identity(nDims))
        elif not isinstance(masses,np.ndarray):
            massArr = masses(path)
        else:
            massArr = masses

        massDim = (nPoints, nDims, nDims)
        if massArr.shape != massDim:
            raise ValueError("Dimension of massArr is "+str(massArr.shape)+\
                             "; required shape is "+str(massDim)+". See action function.")

        if not isinstance(potential,np.ndarray):
            potArr = potential(path)
        else:
            potArr = potential

        potShape = (nPoints,)
        if potArr.shape != potShape:
            raise ValueError("Dimension of potArr is "+str(potArr.shape)+\
                             "; required shape is "+str(potShape)+". See action function.")

        #Actual calculation
        coordDiff = np.diff(path,axis=0)
        actArr = np.einsum("ij,ijk,ik->i",coordDiff,massArr[1:],coordDiff) #The M_{ab} dx^a dx^b bit
        actOut = np.dot(actArr,potArr[1:])
                
        return actOut, potArr, massArr
    
    @staticmethod
    def _term_in_action_squared_sum(points,potential,masses=None):
        """
        One of the discrete terms in the functional computed in "action_squared"
        """
        nDims = points.shape[1]
        if points.shape[0] != 2:
            raise ValueError("Expected exactly 2 points; received "+str(points.shape[0]))

        if masses is None:
            massArr = np.identity(nDims)
        elif not isinstance(masses,np.ndarray):
            massArr = masses(points[1]).reshape((nDims,nDims))
        else:
            massArr = masses

        massDim = (nDims,nDims)
        if massArr.shape != massDim:
            raise ValueError("Dimension of massArr is "+str(massArr.shape)+\
                             "; required shape is "+str(massDim))

        #There's lots of things that can be fed in, including odd things like
        #np.int objects. These aren't caught with a simple filter like "isinstance(potential,int)",
        #so I'm trying it this way for a bit
        try:
            potArr = potential(points[1])
        except TypeError:
            if not isinstance(potential,np.ndarray):
                potArr = np.array([potential])
            else:
                potArr = potential

        potShape = (1,)
        if potArr.shape != potShape:
            raise ValueError("Dimension of potArr is "+str(potArr.shape)+\
                             "; required shape is "+str(potShape))

        #TODO: check if we want this
        potArr = potArr.clip(0)

        #Actual calculation
        coordDiff = points[1] - points[0]
        #The M_{ab} dx^a dx^b bit
        dist = np.dot(coordDiff,np.dot(massArr,coordDiff))
        actOut = 2*potArr[0]*dist

        return actOut, potArr, massArr

    @staticmethod
    def mep_default(points,potential,auxFunc=None):
        '''
        Wrapper to have standard form as other TargetFunctions. For finding the minimum energy path

        Parameters
        ----------
        auxFunc : function or None, optional
            A placeholder function. The default is None
        others : See :py:func:`action`

        Raises
        ------
        ValueError
            See :py:func:`action`

        Returns
        -------
        energies : np.ndarray
            The action along points
        auxEnergies : np.ndarray (None)
            If auxFunc is a function, is auxFunc evaluated at points. Else, is None

        :Maintainer: Eric
        '''
        nPoints, nDim = points.shape
        if not isinstance(potential,np.ndarray):
            potArr = potential(points)
        else:
            potArr = potential
        potShape = (nPoints,)
        if potArr.shape != potShape:
            raise ValueError("Dimension of potArr is "+str(potArr.shape)+\
                             "; required shape is "+str(potShape)+". See potential function.")
        if auxFunc is None:
            auxEnergies = None
        else:
            auxEnergies = auxFunc(points)

        energies  = potential(points)

        return energies, auxEnergies

class GradientApproximations:
    """
    Class containing different gradient approximations for TargetFunctions

    Methods
    -------
    forward_action_grad(path,potential,potentialOnPath,mass,massOnPath,
                        target_func)
        Approximates the gradient of the action with respect to the location of each
        point in path
    forward_action_component_grad(path,potential,potentialOnPath,mass,massOnPath,
                                  target_func)
        Approximates the gradient of each term in the action sum, with respect to the
        location of the rightmost point in the component

    Notes
    -----
    When calling a method of GradientApproximations, we always supply a member of
    TargetFunctions, such as TargetFunctions.action. However, sometimes we
    only want the gradient with respect to one term in the sum that makes up
    target_func. So, we map target_func to a function that evaluates exactly one component
    in the sum. This mapping is defined in GradientApproximations.__init__

    :Maintainer: Daniel
    """
    def __init__(self):
        """
        :Maintainer: Daniel
        """
        self.targetFuncToComponentMap = \
            {"action":TargetFunctions._term_in_action_sum,
             "action_squared":TargetFunctions._term_in_action_squared_sum,
             "mep_default":TargetFunctions.mep_default}

    def discrete_element(self,mass,path,gradOfPes,dr,drp1,beff,beffp1,beffm1,pot,potp1,potm1):
        """
        #TODO: what is this calculating?

        Parameters
        ----------
        mass : function
            Callable mass function
        path : float
            Point i
        gradOfPes : float array
            Gradient of PES at point i
        dr : float array
            dr = r_i - r_i-1
        drp1 : float array
            drp1 = r_i+1 - r_i
        beff : float
            Effective mass at point i
        beffp1 : float
            Effective mass at point i+1
        beffm1 : float
            Effective mass at point i-1
        pot : float
            Potential at point i.
        potp1 : float
            Potential at point i+1
        potm1 : float
            Potential at point i-1

        Returns
        -------
        gradOfAction : float array
            Gradient of action at point i

        :Maintainer: Kyle
        """
        eps = fdTol
        gradOfBeff = self._beff_grad(mass,path,dr,eps=eps)
        dnorm=np.linalg.norm(dr)
        dnormP1=np.linalg.norm(drp1)
        dhat = dr/dnorm
        dhatP1 = drp1/dnormP1
        gradOfAction = 0.5*(\
            (beff*pot + beffm1*potm1)*dhat-\
            (beff*pot + beffp1*potp1)*dhatP1+\
            (beff*gradOfPes + pot*gradOfBeff)*(dnorm+dnormP1))
        return gradOfAction

    def discrete_sqr_action_grad_mp(self,path,potential,potentialOnPath,mass,massOnPath,\
                                    target_func):
        """
        Calculates the analytic form of the discretized gradient of the squared action functional
        named action_squared in Target Functions

        Performs discretized action gradient by parallelizing the gradient calculation
        across NEB images. Still needs numerical PES still


        :Maintainer: Kyle
        """
        eps = fdTol

        gradOfPes = np.zeros(path.shape)
        gradOfBeff = np.zeros(path.shape)
        gradOfAction = np.zeros(path.shape)
        dr = np.zeros(path.shape)
        beff = np.zeros(potentialOnPath.shape)

        nPts, nDims = path.shape

        #Build grad of potential
        gradOfPes = self._midpoint_grad(potential,path,eps=eps)

        dr[1:,:] = np.array([path[ptIter] - path[ptIter-1] \
                               for ptIter in range(1,nPts)])

        beff[1:] = np.array([np.dot(np.dot(massOnPath[ptIter],dr[ptIter]),dr[ptIter])/np.sum(dr[ptIter,:]**2) \
                               for ptIter in range(1,nPts)])
        pool = Pool(helpers.cpu_count())

        mapOut = pool.map(self.discrete_element, \
                itertools.repeat(mass,nPts-1),path[1:nPts-1,:], \
                gradOfPes[1:nPts-1],dr[1:nPts-1,:], dr[2:nPts,:], \
                beff[1:nPts-1], beff[2:nPts], beff[0:nPts-2], \
                potentialOnPath[1:nPts-1], potentialOnPath[2:nPts], potentialOnPath[0:nPts-2])
        mapped = np.array(mapOut)

        gradOfAction[1:nPts-1,:] = mapped[:,0,:]
        return gradOfAction, gradOfPes
    
    def discrete_sqr_action_grad(self,path,potential,potentialOnPath,mass,massOnPath,\
                                     target_func):
        """
        Calculates the analytic form of the discretized gradient of the squared action functional
        named action_squared in Target Functions

        Performs discretized action gradient using a single thread, needs numerical PES still

        :Maintainer: Kyle
        """
        eps = fdTol
        
        potentialOnPath = potentialOnPath[:,None]
        gradOfPes = np.zeros(path.shape)
        gradOfBeff = np.zeros(path.shape)
        gradOfAction = np.zeros(path.shape)
        dr = np.zeros(path.shape)
        beff = np.zeros(potentialOnPath.shape)

        nPts, nDims = path.shape
        
        #Build grad of potential
        gradOfPes = self._midpoint_grad(potential,path,eps=eps)
        
        dr[1:] = np.diff(path,axis=0)
        dnorm = np.linalg.norm(dr,axis=1)
        dhat = dr[1:]/dnorm[1:,None]
        beff[1:,0] = np.einsum("ij,ijk,ik->i",dr[1:],massOnPath[1:],dr[1:])/dnorm[1:]**2
        
        if mass is not None:
            gradOfBeff[1:-1] = self._beff_grad(mass,path[1:-1],dr[1:-1])
                        
        gradOfAction[1:-1] = 0.5*(\
            (beff[1:-1]*potentialOnPath[1:-1] + beff[:-2]*potentialOnPath[:-2])*dhat[:-1]-\
            (beff[1:-1]*potentialOnPath[1:-1] + beff[2:]*potentialOnPath[2:])*dhat[1:]+\
            (beff[1:-1]*gradOfPes[1:-1] + potentialOnPath[1:-1]*gradOfBeff[1:-1])*(dnorm[1:-1,None]+dnorm[2:,None]))
        
        return gradOfAction, gradOfPes
    
    def discrete_action_grad(self,path,potential,potentialOnPath,mass,massOnPath,\
                                 target_func):
        """
        #TODO: what is this calculating?

        Performs discretized action gradient, needs numerical PES still

        :Maintainer: Kyle
        """
        eps = fdTol

        gradOfPes = np.zeros(path.shape)
        gradOfBeff = np.zeros(path.shape)
        gradOfAction = np.zeros(path.shape)
        dr = np.zeros(path.shape)
        beff = np.zeros(potentialOnPath.shape)

        nPts, nDims = path.shape

        #Build grad of potential
        gradOfPes = self._midpoint_grad(potential,path,eps=eps)

        dr[1:,:] = np.array([path[ptIter] - path[ptIter-1] \
                               for ptIter in range(1,nPts)])

        beff[1] = np.dot(np.dot(massOnPath[1],dr[1]),dr[1])/np.sum(dr[1,:]**2)

        for ptIter in range(1,nPts-1):

            gradOfBeff[ptIter] = self._beff_grad(mass,path[ptIter],dr[ptIter],eps=eps)

            beff[ptIter+1] = np.dot(np.dot(massOnPath[ptIter+1],dr[ptIter+1]),dr[ptIter+1])/np.sum(dr[ptIter+1,:]**2)

            dnorm=np.linalg.norm(dr[ptIter])
            dnormP1=np.linalg.norm(dr[ptIter+1])
            dhat = dr[ptIter]/dnorm
            dhatP1 = dr[ptIter+1]/dnormP1
            bv_root = np.sqrt(2.0*beff[ptIter]*potentialOnPath[ptIter])
            bv_rootm1 = np.sqrt(2.0*beff[ptIter-1]*potentialOnPath[ptIter-1])
            bv_rootp1 = np.sqrt(2.0*beff[ptIter+1]*potentialOnPath[ptIter+1])
            gradOfAction[ptIter] = 0.5*(\
                (bv_root + bv_rootm1)*dhat-\
                (bv_root + bv_rootp1)*dhatP1+\
                (beff[ptIter]*gradOfPes[ptIter] + potentialOnPath[ptIter]*gradOfBeff[ptIter])*\
                    (dnorm+dnormP1)/bv_root)

        return gradOfAction, gradOfPes


    def discrete_action_grad_const(self,path,potential,potentialOnPath,mass,massOnPath,\
                            target_func):
        """
        #TODO: what is this calculating?

        Performs discretized action gradient, needs numerical PES still

        :Maintainer: Kyle
        """
        eps = fdTol

        gradOfPes = np.zeros(path.shape)
        gradOfAction = np.zeros(path.shape)

        nPts, nDims = path.shape

        actionOnPath, _, _ = target_func(path,potentialOnPath,massOnPath)

        # build gradOfAction and gradOfPes (constant mass)
        gradOfPes = self._midpoint_grad(potential,path,eps=eps)
        for ptIter in range(1,nPts-1):

            dnorm=np.linalg.norm(path[ptIter] - path[ptIter-1])
            dnormP1=np.linalg.norm(path[ptIter+1] - path[ptIter])
            dhat = (path[ptIter] - path[ptIter-1])/dnorm
            dhatP1 = (path[ptIter+1] - path[ptIter])/dnormP1

            mu=massOnPath[ptIter,0,0]
            gradOfAction[ptIter] = 0.5*(\
                (np.sqrt(2*mu*potentialOnPath[ptIter]) + np.sqrt(2*mu*potentialOnPath[ptIter-1]))*dhat-\
                (np.sqrt(2*mu*potentialOnPath[ptIter]) + np.sqrt(2*mu*potentialOnPath[ptIter+1]))*dhatP1+\
                mu*gradOfPes[ptIter]*(dnorm+dnormP1) / np.sqrt(2*mu*potentialOnPath[ptIter]))

        return gradOfAction, gradOfPes

    def forward_action_grad(self,path,potential,potentialOnPath,mass,massOnPath,\
                            target_func):
        """
        Takes forwards finite difference gradient of any action-like functional

        Parameters
        ----------
        path : np.ndarray
            The path to evaluate the gradient along
        potential : See :py:func:`action`
        potentialOnPath : np.ndarray
            Potential on the path. Of shape (nPoints,).
        mass : See :py:func:`action`
        massOnPath : np.ndarray or None
            Mass on path. If not None, of shape (nPoints,nDims,nDims).
        target_func : function
            Function whose gradient is being computed

        Returns
        -------
        gradOfAction : ndarray
            The gradient of the action
        gradOfPes : ndarray
            The gradient of the energy at each point in path

        Notes
        -----
        The full action is computed at every finite difference step. Does not return the gradient
        of the mass function, as that's not used elsewhere

        :Maintainer: Daniel
        """
        warnings.warn('Deprecation warning: GradientApproximations.forward_action_grad\
                      to be deprecated due to the large number of extra calculations being done')
        eps = fdTol

        gradOfPes = np.zeros(path.shape)
        gradOfAction = np.zeros(path.shape)

        nPts, nDims = path.shape

        actionOnPath, _, _ = target_func(path,potentialOnPath,massOnPath)

        for ptIter in range(nPts):
            for dimIter in range(nDims):
                steps = path.copy()
                steps[ptIter,dimIter] += eps
                actionAtStep, potAtStep, massAtStep = target_func(steps,potential,mass)

                gradOfPes[ptIter,dimIter] = (potAtStep[ptIter] - potentialOnPath[ptIter])/eps
                gradOfAction[ptIter,dimIter] = (actionAtStep - actionOnPath)/eps

        return gradOfAction, gradOfPes

    def forward_action_component_grad(self,path,potential,potentialOnPath,mass,\
                                      massOnPath,target_func):
        """
        Requires an approximation of the action that sums up values along
        the path, such as TargetFunctions.action. Then, this computes the
        forwards finite difference approximation of every term in the sum.

        Note the difference with GradientApproximations().forward_action_grad:
        there, the full action is computed for every step. Here, only the component
        at that step is computed.

        Parameters
        ----------
        target_func : function
            Any term in TargetFunctions that is the sum of some constituent
            terms (e.g. TargetFunctions.action). Uses target_func.__name__
            to select the gradient of a term in the sum, such as
            TargetFunctions._term_in_action_sum
        others : See :py:func:`forward_action_grad`

        Returns
        -------
        See :py:func:`forward_action_grad`

        :Maintainer: Daniel
        """
        targetFuncName = target_func.__name__
        tf_component = self.targetFuncToComponentMap[targetFuncName]

        eps = fdTol

        gradOfPes = np.zeros(path.shape)
        gradOut = np.zeros(path.shape)

        nPts, nDims = path.shape

        if massOnPath is None:
            massOnPath = np.full((nPts,nDims,nDims),np.identity(nDims))

        #Treat the endpoints separately
        for ptIter in range(1,nPts-1):
            for dimIter in range(nDims):
                points = path[ptIter-1:ptIter+1].copy()
                actTermAtPt, _, _ = \
                    tf_component(points,potentialOnPath[ptIter],massOnPath[ptIter])

                points[1,dimIter] += eps
                actTermAtStep, potAtStep, massAtStep = \
                    tf_component(points,potential,mass)

                gradOfPes[ptIter,dimIter] = \
                    (potAtStep - potentialOnPath[ptIter])/eps
                gradOut[ptIter,dimIter] = (actTermAtStep - actTermAtPt)/eps

        #Handling endpoints
        for dimIter in range(nDims):
            pt = path[0].copy()
            pt[dimIter] += eps
            potAtStep = potential(pt)
            gradOfPes[0,dimIter] = (potAtStep - potentialOnPath[0])/eps

            pt = path[-1].copy()
            pt[dimIter] += eps
            potAtStep = potential(pt)
            gradOfPes[-1,dimIter] = (potAtStep - potentialOnPath[-1])/eps

        return gradOut, gradOfPes

    # @numba.jit(forceobj=True)
    def _midpoint_grad(self,func,points,eps=fdTol):
        """
        Midpoint finite difference. Probably best if not used with actual DFT calculations,
            vs a forwards/reverse finite difference
        Assumes func only depends on a single point (vs the action, which depends on
              all of the points)

        :Maintainer: Eric
        """
        if len(points.shape) == 1:
            points = points.reshape((1,-1))
        nPoints, nDims = points.shape
        gradOut = np.zeros((nPoints,nDims))
        for dimIter in range(nDims):
            step = np.zeros(nDims)
            step[dimIter] = 1

            forwardStep = points + eps/2*step
            backwardStep = points - eps/2*step

            forwardEval = func(forwardStep)
            backwardEval = func(backwardStep)

            gradOut[:,dimIter] = (forwardEval-backwardEval)/eps

        return gradOut
    
    def _beff_grad(self,func,points,dr,eps=fdTol):
        """
        Midpoint finite difference of B_eff mass.

        :Maintainer: Kyle
        """
        
        if len(points.shape) == 1:
            points = points.reshape((1,-1))
            #dr = dr.reshape((1,-1))
        nPoints, nDims = points.shape

        gradOut = np.zeros((nPoints,nDims))
        
        ds = np.sum(dr[:]**2,axis=1)
        for dimIter in range(nDims):
            step = np.zeros(nDims)
            step[dimIter] = 1

            forwardStep = points + eps/2*step
            backwardStep = points - eps/2*step

            massP1 = func(forwardStep)
            massM1 = func(backwardStep)
            forwardEval = np.einsum("ij,ijk,ik->i",dr,massP1,dr)/ds
            backwardEval = np.einsum("ij,ijk,ik->i",dr,massM1,dr)/ds

            gradOut[:,dimIter] = (forwardEval-backwardEval)/eps
        return gradOut

    def mep_default(self,points,potential,auxFunc=None):
        '''
        Used in MEP for force updates. There, one only needs the gradient of the
        PES.

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        potential : TYPE
            DESCRIPTION.
        auxFunc : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        gradPES : TYPE
            DESCRIPTION.
        gradAux : TYPE
            DESCRIPTION.

        :Maintainer: Eric
        '''
        gradPES = self._midpoint_grad(potential,points,eps=fdTol)
        if auxFunc is None:
            gradAux = None
        else:
            gradAux = self._midpoint_grad(auxFunc,points,eps=fdTol)
        return gradPES, gradAux

class SurfaceUtils:
    """
    Contains methods for getting information about a surface

    Methods
    -------
    find_all_local_minimum(arr)
        Finds local minima, including edges, of an array
    find_local_minimum(arr,searchPerc,returnOnlySmallest)
        Finds the smallest local minimum in a percent region of arr
    find_approximate_contours(coordMeshTuple,zz,eneg,show)
        Finds contours of energy eneg in D dimensions
    round_points_to_grid(coordMeshTuple,ptsArr)
        Rounds points to their nearest grid location
    find_endpoints_on_grid(coordMeshTuple,potArr,**kwargs)
        Finds contours of a specified energy, and rounds them to
        a grid

    :Maintainer: Daniel
    """
    @staticmethod
    def find_all_local_minimum(arr):
        """
        Returns the indices corresponding to the local minimum values. Taken
        originally from https://stackoverflow.com/a/3986876

        Parameters
        ----------
        arr : np.ndarray
            The array to find minima of. Is D dimensional

        Returns
        -------
        minIndsOut : Tuple of numpy arrays
            D arrays of length k, describing k minima found

        Notes
        -----
        Finder checks along the cardinal directions. If all neighbors in those
        directions are greater than or equal to the current value, the index
        is returned as a minimum. For the border, the array is reflected about
        the axis. As a result, many indices are found that are not technically
        local minima. However, we do want the border results - in practice,
        nuclei often have a ground state at zero deformation in one collective
        coordinate; to find that, we must include the border indices. To exclude
        them, one can then call SurfaceUtils.find_local_minimum
        """
        neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
        #Test case was giving floating-point differences along the outer edge of
        #the array
        local_min = np.isclose(filters.minimum_filter(arr, footprint=neighborhood,\
                                                      mode="nearest"),arr,atol=10**(-15))

        background = (arr==0)
        eroded_background = morphology.binary_erosion(background,\
                                                      structure=neighborhood,\
                                                      border_value=1)

        detected_minima = local_min ^ eroded_background
        allMinInds = np.vstack(local_min.nonzero())
        minIndsOut = tuple([allMinInds[coordIter,:] for \
                            coordIter in range(allMinInds.shape[0])])
        return minIndsOut

    @staticmethod
    def find_local_minimum(arr,searchPerc=[0.25,0.25],returnOnlySmallest=True):
        """
        Returns the indices corresponding to the local minimum values within a
        desired part of the PES

        Parameters
        ----------
        arr : np.ndarray
            The array to find minima of. Is D dimensional
        searchPerc : list, optional
            Percentage of each coordinate that the minimum is allowed to be in.
            The default is [0.25,0.25]
        returnOnlySmallest : bool, optional
            If True, returns only the (first) smallest value. If False, returns
            all minima in the searched region. The default is True

        Raises
        ------
        TypeError
            If arr has a different number of dimensions than len(searchPerc)
        ValueError
            If any of searchPerc are greater than 1

        Returns
        -------
        minIndsOut : Tuple of np.ndarrays
            D arrays of length k, for k minima found in the region. If returnOnlySmallest,
            returns a tuple, not a tuple of arrays

        Notes
        -----
        Note that, if we write searchPerc=[s1,s2], then s1 is the range for
        the first coordinate of arr. If arr was constructed to agree with
        np.meshgrid's default indexing, then s1 will actually restrict the
        range of the second (physical) coordinate: np.meshgrid(X,Y,Z,...)
        returns arrays of shape (Y.len,X.len,Z.len,...)

        """
        if len(searchPerc) != len(arr.shape):
            raise TypeError("searchPerc and arr have unequal lengths ("+\
                            str(len(searchPerc))+") and ("+str(len(arr.shape))+")")

        if np.any(np.array(searchPerc)<=0) or np.any(np.array(searchPerc)>1):
            raise ValueError("All entries in searchPerc must be in the interval (0,1].")

        allMinInds = np.vstack(SurfaceUtils.find_all_local_minimum(arr))

        #Selecting minima within the desired range
        minIndsMask = np.ones(allMinInds.shape[1],dtype=bool)
        for minIter in range(allMinInds.shape[1]):
            for coordIter in range(allMinInds.shape[0]):
                if allMinInds[coordIter,minIter] > \
                    int(searchPerc[coordIter] * arr.shape[coordIter]):
                        minIndsMask[minIter] = False

        allMinInds = allMinInds[:,minIndsMask]

        minIndsOut = tuple([allMinInds[coordIter,:] for \
                            coordIter in range(allMinInds.shape[0])])

        if returnOnlySmallest:
            actualMinInd = np.argmin(arr[minIndsOut])
            minIndsOut = tuple([m[actualMinInd] for m in minIndsOut])

        return minIndsOut

    @staticmethod
    def find_approximate_contours(coordMeshTuple,zz,eneg=0,show=False,
                                  returnAsArr=True):
        """
        Finds 2D contours of energy eneg on a D-dimensional surface

        Parameters
        ----------
        coordMeshTuple : tuple of np.ndarray
            Coordinate mesh, e.g. output of np.meshgrid
        zz : np.ndarray
            Potential on mesh
        eneg : float, optional
            Energy of the desired contour. The default is 0
        show : bool, optional
            Whether to plot the contours. The default is False

        Raises
        ------
        NotImplementedError
            Does not work for 1 dimension.

        Returns
        -------
        allContours : ndarray of lists
            Each element is the returned value of ax.contour.allsegs[0], i.e.
            a list consisting of 2D arrays describing the contour on that slize
            of zz

        Notes
        -----
        Takes 2D surfaces, using the first 2 indices of zz, and iterating over all other
        indices. At every set of indices, pyplot.contour is called, to get the
        2D contour(s) on the surface at that level. The contours are not filled
        with the value of the coordinates with the other indices - i.e. each
        segment is of shape (k,2), regardless of the number of dimensions

        """
        nDims = len(coordMeshTuple)
        uniqueCoords = [np.unique(c) for c in coordMeshTuple]
        coordMeshTuple = np.meshgrid(*uniqueCoords)

        fig, ax = plt.subplots()

        if nDims == 1:
            raise NotImplementedError
        elif nDims == 2:
            allContours = np.zeros(1,dtype=object)
            if show:
                cf = ax.contourf(*coordMeshTuple,zz,cmap="Spectral_r")
                plt.colorbar(cf,ax=ax)
            #Select allsegs[0] b/c I'm only finding one level; ccp.allsegs is a
                #list of lists, whose first index is over the levels requested
            allContours[0] = ax.contour(*coordMeshTuple,zz,levels=[eneg]).allsegs[0]
        else:
            allContours = np.zeros(zz.shape[2:],dtype=object)
            possibleInds = np.indices(zz.shape[2:]).reshape((nDims-2,-1)).T
            for ind in possibleInds:
                meshInds = 2*(slice(None),) + tuple(ind)
                localMesh = (coordMeshTuple[0][meshInds],coordMeshTuple[1][meshInds])
                zzCorrectShape = _get_correct_shape(tuple(np.unique(c) for c in localMesh),
                                                    zz[meshInds])

                contoursOnLevel = ax.contour(*localMesh,zzCorrectShape,levels=[eneg]).allsegs[0]

                levelValues = np.array([np.unique(c[meshInds]) for c in coordMeshTuple[2:]]).flatten()
                contoursWithLevel = []
                for c in contoursOnLevel:
                    cOut = np.zeros((c.shape[0],nDims))
                    cOut[:,:2] = c

                    for (lIter,l) in enumerate(levelValues):
                        cOut[:,lIter+2] = l
                    contoursWithLevel.append(cOut)
                allContours[tuple(ind)] = contoursWithLevel

            if show:
                plt.show(fig)

        if not show:
            plt.close(fig)

        if returnAsArr:
            listOfConts = []
            for c in allContours:
                for subC in c:
                    listOfConts.append(subC)
            allContours = np.vstack(listOfConts)

        return allContours

    @staticmethod
    def round_points_to_grid(coordMeshTuple,ptsArr):
        """
        Rounds an array of points to the nearest point on a grid

        Parameters
        ----------
        coordMeshTuple : tuple of np.ndarrays
            The grid. Taken as output of np.meshgrid
        ptsArr : np.ndarray
            The points to round. Of shape (nPoints,nDims), where nDims is the
            number of coordinates

        Raises
        ------
        NotImplementedError
            Called with nDims < 2
        ValueError
            If ptsArr is the wrong shape

        Returns
        -------
        indsOut : np.ndarray
            The indices of the points. Of shape (nPoints,nDims). See notes
        gridValsOut : np.ndarray
            The nearest grid values. Of shape (nPoints,nDims)

        Notes
        -----
        Has standard complication from np.meshgrid - indexing is (N2,N1,N3,...),
        when the coordinates have lengths (N1,N2,N3,...). This returns the default
        indexing of np.meshgrid for coordMeshTuple. See e.g.
        https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

        """
        nDims = len(coordMeshTuple)

        if nDims < 2:
            raise NotImplementedError("Expected nDims >= 2; recieved "+str(nDims))

        uniqueCoords = [np.unique(c) for c in coordMeshTuple]
        coordMeshTuple = np.meshgrid(*uniqueCoords)

        if ptsArr.shape == (nDims,):
            ptsArr = ptsArr.reshape((1,nDims))

        if ptsArr.shape[1] != nDims:
            raise ValueError("ptsArr.shape = "+str(ptsArr.shape)+\
                             "; second dimension should be nDims, "+str(nDims))

        nPts = ptsArr.shape[0]

        indsOut = np.zeros((nPts,nDims),dtype=int)

        #In case some points are on the grid, which np.searchsorted doesn't seem to
        #handle the way I'd like
        for dimIter in range(nDims):
            for (ptIter, pt) in enumerate(ptsArr[:,dimIter]):
                #Nonsense with floating-point precision makes me use
                #np.isclose rather than a == b
                tentativeInd = np.argwhere(np.isclose(uniqueCoords[dimIter],pt))
                if tentativeInd.shape == (0,1): #Nothing found on grid
                    indsOut[ptIter,dimIter] = \
                        np.searchsorted(uniqueCoords[dimIter],pt) - 1
                else: #Is on grid
                    indsOut[ptIter,dimIter] = tentativeInd

        #I subtract 1 in some cases
        indsOut[indsOut<0] = 0

        gridValsOut = np.zeros((nPts,nDims))
        for ptIter in range(nPts):
            inds = indsOut[ptIter]
            inds[[0,1]] = inds[[1,0]]
            inds = tuple(inds)
            gridValsOut[ptIter] = np.array([c[inds] for c in coordMeshTuple])

        return indsOut, gridValsOut

    @staticmethod
    def find_endpoints_on_grid(coordMeshTuple,potArr,returnAllPoints=False,eneg=0,
                               returnIndices=False):
        """
        Finds points on grid nearest to points of energy eneg

        Parameters
        ----------
        coordMeshTuple : tuple of np.ndarrays
            The grid. Taken as output of np.meshgrid
        potArr : np.ndarray
            The energy on the grid
        returnAllPoints : bool, optional
            Whether to return all indices. The default is False, in which case
            only the connected contour with the most points is returned
        eneg : float, optional
            The energy to end at. The default is 0
        returnIndices : bool, optional
            Whether to return the indices of the gridpoints. The default is False

        Returns
        -------
        allowedEndpoints : np.ndarray
            The allowed endpoints
        allowedIndices : np.ndarray
            The indices for the endpoints. Only returned if returnIndices is True

        """
        if returnAllPoints:
            warnings.warn("find_endpoints_on_grid is finding all "\
                          +"contours; this may include starting point")

        nDims = len(coordMeshTuple)
        uniqueCoords = [np.unique(c) for c in coordMeshTuple]
        coordMeshTuple = np.meshgrid(*uniqueCoords)

        potArr = _get_correct_shape(uniqueCoords,potArr)

        allContours = SurfaceUtils.find_approximate_contours(coordMeshTuple,potArr,
                                                             eneg=eneg,returnAsArr=False)

        allowedEndpoints = np.zeros((0,nDims))
        allowedIndices = np.zeros((0,nDims),dtype=int)

        for contOnLevel in allContours:
            gridContOnLevel = []
            gridIndsOnLevel = []
            for cont in contOnLevel:
                locGridInds, locGridVals = \
                    SurfaceUtils.round_points_to_grid(coordMeshTuple,cont)

                gridIndsOnLevel.append(locGridInds)
                gridContOnLevel.append(locGridVals)

            if returnAllPoints:
                for (cIter,c) in enumerate(gridContOnLevel):
                    allowedEndpoints = np.concatenate((allowedEndpoints,c),axis=0)
                    allowedIndices = np.concatenate((allowedIndices,gridIndsOnLevel[cIter]),axis=0)
            else:
                lenOfContours = np.array([c.shape[0] for c in gridContOnLevel])
                outerIndex = np.argmax(lenOfContours)
                allowedEndpoints = \
                    np.concatenate((allowedEndpoints,gridContOnLevel[outerIndex]),axis=0)
                allowedIndices = \
                    np.concatenate((allowedIndices,gridIndsOnLevel[outerIndex]),axis=0)

        allowedEndpoints = np.unique(allowedEndpoints,axis=0)
        allowedIndices = np.unique(allowedIndices,axis=0)

        if returnIndices:
            return allowedEndpoints, allowedIndices
        else:
            return allowedEndpoints

def shift_func(func_in,shift=10**(-4)):
    """
    Shifts func_in output down by shift. Especially for use with interpolators
    where the minimum of the interpolator may be a bit lower than the minimum of
    the array

    Parameters
    ----------
    func_in : function
    shift : float
        The amount to shift by. The default is 10**(-4).

    Returns
    -------
    func_out : function
        The shifted function

    :Maintainer: Daniel
    """
    def func_out(coords):
        return func_in(coords) - shift
    return func_out

def _get_correct_shape(gridPoints,arrToCheck,normalOrder=True):
    """
    Utility for automatically correcting the shape of an array, to deal with
    nonsense regarding np.meshgrid's default setup
    """
    warnings.warn('DeprecationWarning: Automatic array reshaping will be deprecated soon')

    defaultMeshgridShape = np.array([len(g) for g in gridPoints])
    possibleOtherShape = tuple(defaultMeshgridShape)
    defaultMeshgridShape[[1,0]] = defaultMeshgridShape[[0,1]]
    defaultMeshgridShape = tuple(defaultMeshgridShape)

    if not normalOrder: #Not compliant with np.meshgrid, as in NDInterpWithBoundary for D = 3
        defaultMeshgridShape, possibleOtherShape = possibleOtherShape, defaultMeshgridShape

    isSquare = False
    if defaultMeshgridShape[0] == defaultMeshgridShape[1]:
        isSquare = True
        warnings.warn("Grid is square; cannot check if data is transposed."+\
                      " Note that gridVals should be of shape (x.size,y.size).")

    if arrToCheck.shape == defaultMeshgridShape:
        pass
    elif arrToCheck.shape == possibleOtherShape and not isSquare:
        arrToCheck = np.swapaxes(arrToCheck,0,1)
    else:
        raise ValueError("arrToCheck.shape "+str(arrToCheck.shape)+\
                         " does not match expected shape "+\
                         str(defaultMeshgridShape)+" or possible shape "+\
                         str(possibleOtherShape))

    return arrToCheck
    
class NDInterpWithBoundary:
    """
    Interpolates a grid in D dimensions, with extra handling for points outside
    of the grid. The D>2 case is based on scipy.interpolate.RegularGridInterpolator

    Methods
    -------
    __call__(points)
        Evaluates the interpolator

    :Maintainer: Daniel
    """
    def __init__(self,gridPoints,gridVals,boundaryHandler="exponential",symmExtend=None,\
                 transformFuncName="identity",splKWargs={},
                 _test_linear=False,custom_func=None):
        """
        Parameters
        ----------
        gridPoints : tuple of np.ndarrays
            The unique grid points. Each array must be sorted in ascending order
        gridVals : np.ndarray
            The grid values to be interpolated. Expected to be of shape (N2,N1,N3,...),
            as in the output of np.meshgrid
        boundaryHandler : str, optional
            How points outside of the interpolation region are handled. The
            default is 'exponential'
        symmExtend : None or np.ndarray of bools, optional
            Whether to symmetrically extend gridVals when evaluating. The default is
            None, in which case the second coordinate is symmetrized
        transformFuncName : string, optional
            The function to apply to the interpolated function after interpolating.
            The default is "identity", in which no post-processing is applied
        splKWargs : dict, optional
            Extra arguments for spline interpolation, in the 2D case. The default
            is {}

        Raises
        ------
        NotImplementedError
            Fewer than 2 coordinates
        ValueError
            Any unallowed boundaryHandler or transformFuncName; a wrong-shaped
            array; or an unsorted gridVals

        Notes
        -----
        The boundary handler is assumed to be the same for all dimensions, because
        I can't think of a reasonable way to allow for different handling for
        different dimensions. I also see no reason why one would want to treat
        the dimensions differently.

        Our use case is density functional theory, and our grid points are the
        multipole moments Qi in a constrained DFT calculation. It does not
        always make sense to symmetrically extend a potential energy surface:
        for Q30, it does, while for Q20, it does not. It also does not make sense
        to symmetrically extend the PES at the maximum value. So, symmExtend by
        default assumes Q30 is the second coordinate, and should only be extended
        symmetrically near Q30 = 0; otherwise, everything else is not extended at
        all.

        Also assumes for symmetric extension that the lowest value is 0.

        """
        self.nDims = len(gridPoints)
        if self.nDims < 2:
            raise NotImplementedError("Expected nDims >= 2")

        bdyHandlerFuncs = {"exponential":self._exp_boundary_handler,
                           None:None}
        if boundaryHandler not in bdyHandlerFuncs.keys():
            raise ValueError("boundaryHandler '%s' is not defined" % boundaryHandler)
            
        self.boundaryHandler = bdyHandlerFuncs[boundaryHandler]

        if symmExtend is None:
            symmExtend = np.array([False,True]+(self.nDims-2)*[False],dtype=bool)
        elif not isinstance(symmExtend,np.ndarray):
            warnings.warn("Using symmetric extension "+str(symmExtend)+\
                          " for all dimensions. Make sure this is intended.")
            symmExtend = symmExtend * np.ones(len(gridPoints),dtype=bool)

        if symmExtend.shape != (self.nDims,):
            raise ValueError("symmExtend.shape '"+str(symmExtend.shape)+\
                             "' does not match nDims, "+str(self.nDims))

        self.symmExtend = symmExtend

        self.gridPoints = tuple([np.asarray(p) for p in gridPoints])

        for i, p in enumerate(gridPoints):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
                
        # self.gridVals = gridVals
        if custom_func is None:
            if self.nDims == 2 and not _test_linear:
                self.gridVals = _get_correct_shape(gridPoints,gridVals)
                self.rbv = RectBivariateSpline(*gridPoints,self.gridVals.T,**splKWargs)
                self._call = self._call_2d
            else:
                self.gridVals = _get_correct_shape(gridPoints,gridVals,normalOrder=False)
                #Arguments for scipy.ndimage.map_coordinates
                self.xIntercept = np.array([u[0] for u in self.gridPoints])
                self.dx = np.array([u[1]-u[0] for u in self.gridPoints])
                
                if 'order' in splKWargs:
                    self.splOrder = splKWargs['order']
                else:
                    self.splOrder = 1
                
                self._call = self._call_nd
                if self.nDims == 2:
                    warnings.warn("To use linear interpolation in 2D, pass splKWargs={'kx':1,'ky':1} to initialization")
        else:
            self._call = custom_func

        postEvalDict = {"identity":self._identity_transform_function,
                        "smooth_abs":self._smooth_abs_transform_function}
        self.post_eval = postEvalDict[transformFuncName]
        
    def _rescale_points(self,points):
        return np.moveaxis((points - self.xIntercept)/self.dx,-1,0)
    
    def __call__(self,points):
        """
        Interpolation at points

        Parameters
        ----------
        points : np.ndarray
            The coordinates to sample the gridded data at. Can be more than 2D,
            as in points.shape == complexShape + (self.nDims,)

        Returns
        -------
        result : np.ndarray
            The interpolated function evaluated at points. Is of shape complexShape

        """
        originalShape = points.shape[:-1]
        if originalShape == ():
            originalShape = (1,)

        if points.shape[-1] != self.nDims:
            raise ValueError("The requested sample points have dimension "
                             "%d, but this NDInterpWithBoundary expects "
                             "dimension %d" % (points.shape[-1], self.nDims))

        points = points.reshape((-1,self.nDims))

        #Dealing with symmetric extension
        for dimIter in range(self.nDims):
            if self.symmExtend[dimIter]:
                points[:,dimIter] = np.abs(points[:,dimIter])
        
        if self.boundaryHandler is None:
            res = self._call(points)
        else:
            #Checking if each point is acceptable, and interpolating individual points.
            evalLoc = points.copy()
            
            for dimIter in range(self.nDims):
                #Changed from np.clip due to https://github.com/numpy/numpy/issues/14281
                evalLoc[:,dimIter] = np.core.umath.clip(evalLoc[:,dimIter],
                                                        self.gridPoints[dimIter][0],
                                                        self.gridPoints[dimIter][-1])
                
            res = self._call(evalLoc)
            res *= self.boundaryHandler(evalLoc,points)
        
        res = self.post_eval(res)

        return res.reshape(originalShape)

    def _call_2d(self,points):
        """
        Evaluates the RectBivariateSpline instance at many points. Defined
        as a wrapper here so that self._call has the same calling signature
        regardless of dimension
        """
        return self.rbv(points[:,0],points[:,1],grid=False)

    # def _call_nd_old(self,points):
    #     """
    #     Repeated linear interpolation. For the 2D case, see e.g.
    #     https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean

    #     Notes
    #     -----
    #     Original implementation, taken from scipy.interpolate.RegularGridInterpolator,
    #     handled multiple points at a time. I've trimmed things down here so that
    #     it only handles a single point at a time, since the loop in self.__call__
    #     has to check every point individually anyways.

    #     """
    #     indices, normDistances = self._find_indices(points.T)
    #     # print(indices)
    #     # print(normDistances)
        
    #     vals = np.zeros(points.shape[0])
    #     for ptIter in range(points.shape[0]):
    #         # find relevant values
    #         # each i and i+1 represents a edge
    #         edges = itertools.product(*[[i, i + 1] for i in indices[:,ptIter]])
    #         for edge_indices in edges:
    #             weight = 1.
    #             for ei, i, yi in zip(edge_indices, indices[:,ptIter], normDistances[:,ptIter]):
    #                 weight *= np.where(ei == i, 1 - yi, yi).item()
    #             vals[ptIter] += weight * self.gridVals[edge_indices].item()
        
    #     return vals
    
    def _call_nd(self,points):
        return map_coordinates(self.gridVals,
                               self._rescale_points(points),
                               order=self.splOrder,mode='nearest')

    # def _find_indices(self,points):
    #     """
    #     Finds indices of nearest gridpoint, utilizing the regularity of the grid.
    #     Also computes how far in each coordinate dimension every point is from
    #     the previous gridpoint (not the nearest), normalized such that the next
    #     gridpoint (in a particular dimension) is distance 1 from the nearest gridpoint
    #     (called unity units). The distance is normed to make the interpolation
    #     simpler

    #     Taken from scipy.interpolate.RegularGridInterpolator.

    #     Example
    #     -------
    #     Returned indices of ([2,3],[1,7],[3,2]) indicates that the first point
    #     has nearest grid index (2,1,3), and the second has nearest grid index
    #     (3,7,2).

    #     Notes
    #     -----
    #     If the nearest edge is the outermost edge in a given coordinate, the inner
    #     edge is return instead.

    #     Requires points to have first dimension equal to self.nDims so that
    #     this can zip points and self.gridPoints

    #     """
    #     indices = []
    #     normDistances = np.zeros(points.shape)

    #     for (coordIter,x,grid) in zip(np.arange(self.nDims),points,self.gridPoints):
    #         #This is why the grid must be sorted - this search is now quick. All
    #         #this does is find the index in which to place x such that the list
    #         #self.grid[coordIter] remains sorted.
    #         i = np.searchsorted(grid, x) - 1
            
    #         #If x would be the new first element, index it as zero
    #         i[i < 0] = 0
    #         #If x would be the last element, make it not so. However, the way
    #         #the interpolation scheme is set up, we need the nearest gridpoint
    #         #that is not the outermost gridpoint. See below with grid[i+1]
    #         i[i > grid.size - 2] = grid.size - 2

    #         indices.append(i)
    #         normDistances[coordIter] = (x - grid[i]) / (grid[i + 1] - grid[i])

    #     return np.array(indices), normDistances
    
    def _exp_boundary_handler(self,evalLoc,points):
        """
        Computes exp(sqrt(dist)) between the desired evaluation location, points,
        and the nearest points inside of the grid, evalLoc
        """
        diff = evalLoc - points
        dist = np.linalg.norm(diff,axis=1)
        return np.exp(np.sqrt(dist))

    def _identity_transform_function(self,normalEvaluation):
        """
        Returns normal evaluation. Not sure if it's faster to have this dummy
        function in place, or to have an "if-else" statement every time we check
        if we should call a transform function
        """
        return normalEvaluation

    def _smooth_abs_transform_function(self,normalEvaluation):
        """
        A smooth approximation of the absolute value of the energy
        """
        return np.sqrt(normalEvaluation**2 + 10**(-8))

class PositiveSemidefInterpolator:
    """
    Interpolates a positive semidefinite function in D dimensions. Takes
    eigenvalue decomposition, interpolates eigenvalues and unique components
    of eigenvectors using NDInterpWithBoundary, and truncates eigenvalues at 0

    Methods
    -------
    __call__(points)
        Evaluates the interpolators

    Notes
    -----
    Interpolator for D > 2 is not positive semidefinite, although it is
    symmetric. Is intended to be PSD, sometime in the future
    """
    def __init__(self,gridPoints,listOfVals,customInterp=None,ndInterpKWargs={},_test_nd=False,
                 ):
        """
        Parameters
        ----------
        gridPoints : tuple
            Elements are the unique coordinates of the grid. Shapes are
                (N1,N2,...,Nn).
        listOfVals : list
            A positive semidefinite matrix M has unique values
                M = [[M00, M01, ..., M0n],
                     [M01, M11, ..., M1n],
                     ...,
                     [M0n, M1n, ..., Mnn]].
            The components of listOfVals are the numpy arrays
                [M00, M01, ..., M0n, M11, M12, ..., M1n, ..., Mnn].
            Each Mij is of shape (N2,N1,N3,...), as in the output of np.meshgrid.
        ndInterpKWargs : dict, optional
            Options to feed into NDInterpWithBoundary. The default is {}

        Raises
        ------
        NotImplementedError
            If D != 2
        """
        self.nDims = len(gridPoints)
        self.gridPoints = gridPoints
        self.ndInterpKWargs = ndInterpKWargs
        self.customInterp = customInterp

        #Standard error checking
        assert len(listOfVals) == int(self.nDims*(self.nDims+1)/2)
        
        # self.gridValsList = listOfVals
        if self.nDims == 2 and _test_nd == False:
            self.gridValsList = [_get_correct_shape(gridPoints,l) for l in listOfVals]
            self._construct_interps_2d()
            self._call = self._call_2d
        elif self.nDims > 2 or _test_nd:
            warnings.warn("Interpolation for D > 2 not positive semidefinite")
            self.gridValsList = [_get_correct_shape(gridPoints,l,normalOrder=False) for l in listOfVals]
            self._construct_interps_nd()
            self._call = self._call_nd
            if self.nDims == 2:
                warnings.warn("To use linear interpolation in 2D, pass ndInterpKWargs={'splKWargs':{'kx':1,'ky':1}} to initialization")
        else:
            raise NotImplementedError

    def _construct_interps_2d(self):
        """
        Takes eigen decomposition of matrix, and interpolates the eigenvalues
        and unique component of the eigenvectors using NDInterpWithBoundary
        
        TODO: set up multiple different interpolators for different components
        """
        #Taking shortcuts because I only care about D=2 right now
        self.gridVals = np.stack((np.stack((self.gridValsList[0],self.gridValsList[1])),\
                                  np.stack((self.gridValsList[1],self.gridValsList[2]))))
        self.gridVals = np.moveaxis(self.gridVals,[0,1],[2,3])

        eigenVals, eigenVecs = np.linalg.eigh(self.gridVals)
        
        #Constructing interpolators
        if self.customInterp is None:
            self._eigenValInterps = [NDInterpWithBoundary(self.gridPoints,e.T,**self.ndInterpKWargs)\
                                     for e in eigenVals.T]
            self._eigenVecInterp = NDInterpWithBoundary(self.gridPoints,eigenVecs[:,:,0,0],**self.ndInterpKWargs)
        else:
            if self.ndInterpKWargs:
                warnings.warn('ndInterpKWargs has arguments provided but not used')
            self._eigenValInterps = [self.customInterp(self.gridPoints,e)\
                                     for e in eigenVals.T]
            self._eigenVecInterp = self.customInterp(self.gridPoints,eigenVecs[:,:,0,0])

        return None

    def _construct_interps_nd(self):
        """
        Interpolates unique components of the inertia using NDInterpWithBoundary
        """
        if self.customInterp is None:
            self._componentInterps = [NDInterpWithBoundary(self.gridPoints,m,**self.ndInterpKWargs)\
                                      for m in self.gridValsList]
        else:
            if self.ndInterpKWargs:
                warnings.warn('ndInterpKWargs has arguments provided but not used')
            self._componentInterps = [self.customInterp(self.gridPoints,m)\
                                      for m in self.gridValsList]

        return None
    # @profile
    def _call_2d(self,points):
        """
        Evaluates the interpolator in 2 dimensions
        """
        originalShape = points.shape[:-1]
        if originalShape == ():
            originalShape = (1,)

        if points.shape[-1] != self.nDims:
            raise ValueError("The requested sample points have dimension "
                             "%d, but this interpolator expects "
                             "dimension %d" % (points.shape[-1], self.nDims))

        eigenVals = [e(points) for e in self._eigenValInterps]
        #Noticeably faster than arr.clip(0,1)
        ct = np.core.umath.clip(self._eigenVecInterp(points),-1,1)
        # ct = self._eigenVecInterp(points).clip(0,1)
        st = np.sqrt(1-ct**2)
        
        #Noticeably faster that e.clip(0), for some reason
        eigenVals = [np.core.umath.clip(e,0,e.max()) for e in eigenVals]

        ret = np.zeros(originalShape+(2,2))
        ndims = len(points.shape)

        ret[(ndims-1)*(slice(None),)+(0,0)] = eigenVals[0]*ct**2 + eigenVals[1]*st**2
        ret[(ndims-1)*(slice(None),)+(0,1)] = (eigenVals[0]-eigenVals[1])*st*ct
        ret[(ndims-1)*(slice(None),)+(1,0)] = ret[(ndims-1)*(slice(None),)+(0,1)]
        ret[(ndims-1)*(slice(None),)+(1,1)] = eigenVals[0]*st**2 + eigenVals[1]*ct**2
                
        return ret

    def _call_nd(self,points):
        """
        Evaluates the interpolator in D>2 dimensions
        """
        evaluatedComponents = [f(points) for f in self._componentInterps]

        ret = np.zeros(points.shape[:-1]+(self.nDims,self.nDims))

        upperTriInds = np.triu_indices(self.nDims)

        sliceTuple = (ret.ndim-2)*(slice(None),)
        for (evalIter,evalArr) in enumerate(evaluatedComponents):
            rowIter = upperTriInds[0][evalIter]
            colIter = upperTriInds[1][evalIter]

            ret[sliceTuple+(rowIter,colIter)] = evalArr
            #Symmetrizing
            ret[sliceTuple+(colIter,rowIter)] = evalArr

        return ret

    def __call__(self,points):
        """
        Evaluates the matrix at points

        Parameters
        ----------
        points : np.ndarray
            The evaluation location

        Returns
        -------
        np.ndarray
            The evaluated array

        """
        return self._call(points)

class InterpolatedPath:
    """
    Interpolates along a path, with unit path length

    Methods
    -------
    __call__(t)
        Evaluates the path at some point t in [0,1]
    compute_along_path(target_func,nImages,tfArgs,tfKWargs)
        Evaluates a functional along the path, with a specific number
        of interpolation points along the path

    :Maintainer: Daniel
    """
    def __init__(self,discretePath,kwargs={}):
        """
        Parameters
        ----------
        discretePath : np.ndarray
            The path to interpolate
        kwargs : dict, optional
            The keyword arguments for scipy's splprep. The default is {}, in
            which case the defaults are taken from defaultKWargs

        Notes
        -----
        Note that when one considers a 1D curve embedded in 2D, e.g. in a plot
        of a function, one should specify 'u' in kwargs. Otherwise, 'u' will
        be computed based on the distance between points on the path, which
        will generally lead to a different plot than what is desired
        """
        self.path = discretePath

        defaultKWargs = {"full_output":True,"s":0,"k":1}
        for arg in defaultKWargs:
            if arg not in kwargs:
                kwargs[arg] = defaultKWargs[arg]

        if self.path.ndim == 1:
            listOfCoords = [self.path]
        else:
            listOfCoords = [d for d in self.path.T]

        if kwargs["full_output"]:
            (self.tck, self.u), self.fp, self.ier, self.msg = \
                splprep(listOfCoords,**kwargs)
            # print(self.msg)
        else:
            self.tck, self.u = splprep(listOfCoords,**kwargs)

    def __call__(self,t):
        """
        Evaluates the path at a point t in [0,1]

        Parameters
        ----------
        t : float
            The point on the arc length

        Raises
        ------
        ValueError
            If t is not in [0,1]

        Returns
        -------
        np.ndarray
            The points at t
        """
        if np.any(t>1) or np.any(t<0):
            raise ValueError("t must be between 0 and 1 for interpolation")

        return splev(t,self.tck)

    def compute_along_path(self,target_func,nImages,tfArgs,tfKWargs):
        """
        Computes a functional, from TargetFunctions, along the path

        Parameters
        ----------
        target_func : function
            The functional to evaluate
        nImages : int
            The number of sample points along the path to use when evaluating
            target_func
        tfArgs : list
            The arguments for target_func. See :py:func:`action`
        tfKWargs : dict
            The keyword arguments for target_func. See :py:func:`action`

        """
        t = np.linspace(0,1,nImages)
        path = np.array(self.__call__(t)).T
        tfOut = target_func(path,*tfArgs,**tfKWargs)

        return path, tfOut
    
    def find_points_with_select_energy(self,pes,wantedEneg,enegThresh=0.05,nImages=500):
        t = np.linspace(0,1,nImages)
        path = np.array(self.__call__(t)).T
        enegOnPath = pes(path)
        
        matchInds = np.where(np.abs(enegOnPath - wantedEneg) < enegThresh)[0]
        return path[matchInds]

def get_crit_pnts(V_func,path,method='central'):
    '''
    WARNING: This function depends on a package called autograd for hessian calculation
    When using this function, you need to import numpy using: import autograd.numpy as np

    This function finds the critical the MEP path must pass through by first finding the
    critical points of the potential function evaluated along the curve and then classifies
    using the eigenvalues of the Hessian. Returns minima, maxima, and saddle points indices
    along the path.

    Parameters
    ----------
    V_func : object
        Energy Function that must have shape (nImgs,nDims).
    path : ndarray
        coordinates of the path on the surface with shape (nImgs,nDims).
    method : string
        differentiation method option for numdifftools. Options are
        ‘central’, ‘complex’, ‘multicomplex’, ‘forward’, ‘backward’. See
        https://numdifftools.readthedocs.io/en/latest/reference

    Returns
    -------
    3 arrays containing the indices of minima, maxima, and saddle points.

    '''
    ### path should be shape (nImgs,nDims)
    nDim = path.shape[1]
    EnergyOnPath = np.zeros(path.shape)
    for i,pnt in enumerate(path):
        EnergyOnPath[i] = V_func(pnt)
    minima_pnts = SurfaceUtils.find_all_local_minimum(EnergyOnPath)[0]
    maxima_pnts = SurfaceUtils.find_all_local_minimum(-EnergyOnPath)[0]
    crit_pnts = np.concatenate((minima_pnts,maxima_pnts))
    maxima = []
    minima = []
    saddle = []
    for indx in crit_pnts:
        coord = path[indx]
        hess = nd.Hessian(V_func,method=method)(coord)
        evals = np.linalg.eigvals(hess)
        for j,val in enumerate(evals):
            if abs(val) < 10**(-6):
                evals[j] = 0
        ## see which components are less than 0.

        neg_bool = evals < 0
        ## count how many false vals there are (ie how many postives evals there are)
        eval_num = np.count_nonzero(neg_bool)
        if eval_num == 0:
            # if all evals are positive, then H is positive def and the function is
            # concave up at this point. This means we are at a local minima
            minima.append(indx)
        elif eval_num == nDim:
            # if all evals are positive, then H is negative def and the function is
            # concave down at this point. This means we are at a local maxima
            maxima.append(indx)
        else:
            # if evals are positive and negative,
            # this means we are at a local saddle
            saddle.append(indx)
        ## stupid way of removing duplicate indicies
        maxima = list(set(maxima))
        minima = list(set(minima))
        saddle = list(set(saddle))
    return(maxima,minima,saddle)