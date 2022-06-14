from fileio import *

import numpy as np
#import numdifftools as nd
import numdifftools as nd
import sys
import matplotlib.pyplot as plt
import itertools

from scipy.interpolate import interpnd, RectBivariateSpline, splprep, splev
from scipy.ndimage import filters, morphology #For minimum finding
from pathos.multiprocessing import ProcessingPool as Pool
import warnings

global fdTol
fdTol = 10**(-8)

class TargetFunctions:
    #No need to do any compatibility checking with gradients here.
    @staticmethod
    def action(path,potential,masses=None):
        """
        
        TODO: docs
        Allowed masses:
            -Constant mass; set masses = None
            -Array of values; set masses to a numpy array of shape (nPoints, nDims, nDims)
            -A function; set masses to a function
        Allowed potential:
            -Array of values; set potential to a numpy array of shape (nPoints,)
            -A function; set masses to a function
            
        Computes action as
            $ S = sum_{i=1}^{nPoints} sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
            
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
        potArr = potArr.clip(0)
            
        #Actual calculation
        actOut = 0
        for ptIter in range(1,nPoints):
            coordDiff = path[ptIter] - path[ptIter - 1]
            dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
            if dist<0:
                dist = 0
            #     print(dist)
            #     print(path)
            #     print(potArr)
            #     print(massArr)
            actOut += np.sqrt(2*potArr[ptIter]*dist)
        
        return actOut, potArr, massArr
    
    @staticmethod
    def term_in_action_sum(points,potential,masses=None):
        """
        
        TODO: docs
        Allowed masses:
            -Constant mass; set masses = None
            -Array of values; set masses to a numpy array of shape (nPoints, nDims, nDims)
            -A function; set masses to a function
        Allowed potential:
            -Array of values; set potential to a numpy array of shape (nPoints,)
            -A function; set masses to a function
            
        Computes action as
            $ S = sum_{i=1}^{nPoints} sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
            
        :Maintainer: Daniel
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
        actOut = np.sqrt(2*potArr[0]*dist)
        
        return actOut, potArr, massArr
    
    @staticmethod
    def action_squared(path,potential,masses=None):
        '''
        
        Parameters
        ----------
        path : ndarray
            np.ndarray of shape (Nimgs,nDim) containing postions of all images.
        potential : object or ndarray
            Allowed potential:
            -Array of values; set potential to a numpy array of shape (nPoints,)
            -A function; set masses to a function
        masses : object or ndarray, Optional
            Allowed masses:
            -Constant mass; set masses = None
            -Array of values; set masses to a numpy array of shape (nPoints, nDims, nDims)
            -A function; set masses to a function

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        actOut : float
            
        potArr : ndarray
            ndarray of shape (Nimgs,1) containing the PES values for each image in path
        massArr : ndarray
            ndarray of shape (Nimgs,nDim,nDim) containing the mass tensors for each image in path.
            
        :Maintainer: Eric
        '''
        """    
        Computes action as
            $ S = sum_{i=1}^{nPoints} E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b $
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
            
        #Actual calculation
        actOut = 0
        for ptIter in range(1,nPoints):
            coordDiff = path[ptIter] - path[ptIter - 1]
            dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
            actOut += potArr[ptIter]*dist
        return actOut, potArr, massArr
    
    @staticmethod
    def term_in_action_squared_sum(points,potential,masses=None):
        """
        
        TODO: docs
        Allowed masses:
            -Constant mass; set masses = None
            -Array of values; set masses to a numpy array of shape (nPoints, nDims, nDims)
            -A function; set masses to a function
        Allowed potential:
            -Array of values; set potential to a numpy array of shape (nPoints,)
            -A function; set masses to a function
            
        Computes action as
            $ S = sum_{i=1}^{nPoints} sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
            
        :Maintainer: Daniel
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
        
        Essentially a wrapper function for the potential. Expected points to be 
        a (nPts,nDim) matrix. Potential should be a function capable of returning 
        a (nPts,nDim) matrix.

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        potential : TYPE
            DESCRIPTION.
        auxFunc : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        energies : TYPE
            DESCRIPTION.
        auxEnergies : TYPE
            DESCRIPTION.
            
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
    def __init__(self):
        """
        
        When calling a method of GradientApproximations, we always supply a
        target_func, such as TargetFunctions.action. However, sometimes we
        only want the gradient wrt one term in the sum that makes up target_func.
        So, we map target_func to a function that evaluates exactly one component
        in the sum. This mapping is defined here.

        Returns
        -------
        None.
        
        :Maintainer: Daniel
        """
        self.targetFuncToComponentMap = \
            {"action":TargetFunctions.term_in_action_sum,
             "action_squared":TargetFunctions.term_in_action_squared_sum}
    
    def discrete_element(self,mass,path,gradOfPes,dr,drp1,beff,beffp1,beffm1,pot,potp1,potm1):
        """
        

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
        gradOfBeff = beff_grad(mass,path,dr,eps=eps)
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
        gradOfPes = midpoint_grad(potential,path,eps=eps)

        dr[1:,:] = np.array([path[ptIter] - path[ptIter-1] \
                               for ptIter in range(1,nPts)])
        
        beff[1:] = np.array([np.dot(np.dot(massOnPath[ptIter],dr[ptIter]),dr[ptIter])/np.sum(dr[ptIter,:]**2) \
                               for ptIter in range(1,nPts)])
        pool = Pool(6)

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
        gradOfPes = midpoint_grad(potential,path,eps=eps)
        
        dr[1:,:] = np.array([path[ptIter] - path[ptIter-1] \
                               for ptIter in range(1,nPts)])

        beff[1] = np.dot(np.dot(massOnPath[1],dr[1]),dr[1])/np.sum(dr[1,:]**2)
        
        if mass is not None:
            for ptIter in range(1,nPts-1):
                gradOfBeff[ptIter] = beff_grad(mass,path[ptIter],dr[ptIter],eps=eps)
        
        for ptIter in range(1,nPts-1):
            beff[ptIter+1] = np.dot(np.dot(massOnPath[ptIter+1],dr[ptIter+1]),dr[ptIter+1])/np.sum(dr[ptIter+1,:]**2)
            
            dnorm=np.linalg.norm(dr[ptIter])
            dnormP1=np.linalg.norm(dr[ptIter+1])
            dhat = dr[ptIter]/dnorm
            dhatP1 = dr[ptIter+1]/dnormP1

            gradOfAction[ptIter] = 0.5*(\
                (beff[ptIter]*potentialOnPath[ptIter] + beff[ptIter-1]*potentialOnPath[ptIter-1])*dhat-\
                (beff[ptIter]*potentialOnPath[ptIter] + beff[ptIter+1]*potentialOnPath[ptIter+1])*dhatP1+\
                (beff[ptIter]*gradOfPes[ptIter] + potentialOnPath[ptIter]*gradOfBeff[ptIter])*(dnorm+dnormP1))
        
        return gradOfAction, gradOfPes
    
    def discrete_action_grad(self,path,potential,potentialOnPath,mass,massOnPath,\
                                 target_func):
        """

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
        gradOfPes = midpoint_grad(potential,path,eps=eps)

        dr[1:,:] = np.array([path[ptIter] - path[ptIter-1] \
                               for ptIter in range(1,nPts)])

        beff[1] = np.dot(np.dot(massOnPath[1],dr[1]),dr[1])/np.sum(dr[1,:]**2)

        for ptIter in range(1,nPts-1):

            gradOfBeff[ptIter] = beff_grad(mass,path[ptIter],dr[ptIter],eps=eps)

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
        
        Performs discretized action gradient, needs numerical PES still
        
        :Maintainer: Kyle
        """
        eps = fdTol
        
        gradOfPes = np.zeros(path.shape)
        gradOfAction = np.zeros(path.shape)
        
        nPts, nDims = path.shape
        
        actionOnPath, _, _ = target_func(path,potentialOnPath,massOnPath)

        # build gradOfAction and gradOfPes (constant mass)
        gradOfPes = midpoint_grad(potential,path,eps=eps)
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
        Takes forwards finite difference approx of any action-like function.
        See e.g. TargetFunctions.action. Note that the full action is computed
        at every finite difference step.
        
        Does not return the gradient of the mass function, as that's not used 
        elsewhere.

        Parameters
        ----------
        path : ndarray
            The path. Of shape (nPoints,nDims)
        potential : -
            As allowed in TargetFunctions.action
        potentialOnPath : ndarray
            Potential on the path. Of shape (nPoints,).
        mass : -
            As allowed in TargetFunctions.action
        massOnPath : ndarray or None
            Mass on path. If not None, of shape (nPoints,nDims,nDims).
        target_func : function
            Function whose gradient is being computed

        Returns
        -------
        gradOfAction : ndarray
        gradOfPes : ndarray
        
        :Maintainer: Daniel
        """
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
        Requires an approximation of the action that just sums up values along
        the path, such as TargetFunctions.action. Then, this computes the
        forwards finite difference approximation of every *term in the sum*.
        
        Note the difference with GradientApproximations().forward_action_grad:
        there, the full action is computed for every step. Here, only the component
        at that step is computed.
        
        Does not return the gradient of the mass function, as that's not used 
        elsewhere.

        Parameters
        ----------
        path : ndarray
            The path. Of shape (nPoints,nDims)
        potential : function.
            Must take as input an array of shape path.shape
        potentialOnPath : ndarray
            Potential on the path. Of shape (nPoints,).
        mass : function or None
        massOnPath : ndarray or None
            Mass on path. If not None, of shape (nPoints,nDims,nDims).
        target_func : function
            Any term in TargetFunctions that is the sum of some constituent
            terms (e.g. TargetFunctions.action). Uses target_func.__name__
            to select the gradient of a term in the sum, such as 
            TargetFunctions.term_in_action_sum

        Returns
        -------
        gradOfAction : ndarray
        gradOfPes : ndarray
        
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

def potential_central_grad(points,potential,auxFunc=None):
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
    h = 10**(-8)
    ## check if it is a scalar is done inside midpoint_grad
    gradPES = midpoint_grad(potential,points,eps=h)
    if auxFunc is None:
        gradAux = None
    else: 
        gradAux = midpoint_grad(auxFunc,points,eps=h)
    return gradPES, gradAux

def midpoint_grad(func,points,eps=10**(-8)):
    """
    TODO: allow for arbitrary shaped outputs, for use with inertia tensor
    TODO: maybe only have one gradient approx ever
    
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

def beff_grad(func,points,dr,eps=10**(-8)):
    """
    Midpoint finite difference of B_eff mass.
    
    :Maintainer: Kyle
    """
    if len(points.shape) == 1:
        points = points.reshape((1,-1))
        #dr = dr.reshape((1,-1))
    nPoints, nDims = points.shape

    gradOut = np.zeros((nPoints,nDims))

    ds = np.sum(dr[:]**2)

    for dimIter in range(nDims):
        step = np.zeros(nDims)
        step[dimIter] = 1

        forwardStep = points + eps/2*step
        backwardStep = points - eps/2*step

        massP1 = func(forwardStep)
        massM1 = func(backwardStep)

        forwardEval = np.dot(np.dot(massP1,dr),dr)/ds
        backwardEval = np.dot(np.dot(massM1,dr),dr)/ds

        gradOut[:,dimIter] = (forwardEval-backwardEval)/eps 
    return gradOut


class SurfaceUtils:
    """
    Defined for namespace purposes
    
    :Maintainer: Daniel
    """
    @staticmethod
    def find_all_local_minimum(arr):
        """
        Returns the indices corresponding to the local minimum values. Taken
        originally from https://stackoverflow.com/a/3986876
        
        Finder checks along the cardinal directions. If all neighbors in those
        directions are greater than or equal to the current value, the index
        is returned as a minimum. For the border, the array is reflected about
        the axis. As a result, many indices are found that are not technically
        local minima. However, we do want the border results - in practice,
        nuclei often have a ground state at zero deformation in one collective
        coordinate; to find that, we must include the border indices. To exclude
        them, one can then call SurfaceUtils.find_local_minimum.
        
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
    
    def find_all_local_maximum(arr):
        """
        Returns the indices corresponding to the local maximum values. Taken
        originally from https://stackoverflow.com/a/3986876
        
        Finder checks along the cardinal directions. If all neighbors in those
        directions are greater than or equal to the current value, the index
        is returned as a minimum. For the border, the array is reflected about
        the axis. As a result, many indices are found that are not technically
        local minima. However, we do want the border results - in practice,
        nuclei often have a ground state at zero deformation in one collective
        coordinate; to find that, we must include the border indices. 
        
        Parameters
        ----------
        arr : Numpy array
            A D-dimensional array.
    
        Returns
        -------
        maxIndsOut : Tuple of numpy arrays
            D arrays of length k, for k maxima found
    
        """
        neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
        #Test case was giving floating-point differences along the outer edge of
        #the array
        local_max = np.isclose(filters.maximum_filter(arr, footprint=neighborhood,\
                                                      mode="nearest"),arr,atol=10**(-15))
        
        background = (arr==0)
        eroded_background = morphology.binary_erosion(background,\
                                                      structure=neighborhood,\
                                                      border_value=1)
            
        detected_maxima = local_max ^ eroded_background
        allMaxInds = np.vstack(local_max.nonzero())
        maxIndsOut = tuple([allMaxInds[coordIter,:] for \
                            coordIter in range(allMaxInds.shape[0])])
        return maxIndsOut
    
    def find_local_minimum(arr,searchPerc=[0.25,0.25],returnOnlySmallest=True):
        """
        Returns the indices corresponding to the local minimum values within a
        desired part of the PES.
        
        Parameters
        ----------
        arr : Numpy array
            A D-dimensional array.
        searchPerc : List
            Percentage of each coordinate that the minimum is allowed to be in.
            See Notes for a note on searchPerc
        returnOnlySmallest : Bool. Default is True
            If True, returns only the (first) smallest value. If False, returns
            all minima in the searched region.
    
        Returns
        -------
        minIndsOut : Tuple of numpy arrays
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
    def find_approximate_contours(coordMeshTuple,zz,eneg=0,show=False):
        """
        Finds 2D contours on a D-dimensional surface. Does so by considering
        2D surfaces, using the first 2 indices of zz, and iterating over all other
        indices. At every set of indices, pyplot.contour is called, to get the
        2D contour(s) on the surface at that level. The contours are not filled
        with the value of the coordinates with the other indices - i.e. each
        segment is of shape (k,2), regardless of the number of dimensions.

        Parameters
        ----------
        coordMeshTuple : tuple of ndarray
            Coordinate mesh, e.g. output of np.meshgrid
        zz : ndarray
            Potential on mesh
        eneg : float, optional
            Energy of the desired contour. The default is 0.
        show : bool, optional
            Whether to plot the contours. The default is False.

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

        """
        nDims = len(coordMeshTuple)
        
        fig, ax = plt.subplots()
        
        if nDims == 1:
            raise NotImplementedError("Why are you looking at D=1?")
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
                # print(localMesh)
                allContours[tuple(ind)] = \
                    ax.contour(*localMesh,zz[meshInds],levels=[eneg]).allsegs[0]
            if show:
                plt.show(fig)
                
        if not show:
            plt.close(fig)
        
        return allContours
    
    @staticmethod
    def round_points_to_grid(coordMeshTuple,ptsArr):
        """
        Rounds an array of points to the nearest point on a grid.

        Parameters
        ----------
        coordMeshTuple : tuple of ndarrays
            The grid. Taken as output of np.meshgrid
        ptsArr : ndarray
            The points to round. Of shape (nPoints,nDims), where nDims is the
            number of coordinates.

        Returns
        -------
        indsOut : ndarray of ints
            The indices of the points. Of shape (nPoints,nDims). See notes.
        gridValsOut : ndarray
            The nearest grid values. Of shape (nPoints,nDims).
        
        Notes
        -----
        Has standard complication from np.meshgrid - indexing is (N2,N1,N3,...),
        when the coordinates have lengths (N1,N2,N3,...). This returns the default
        indexing of np.meshgrid for coordMeshTuple. See e.g. 
        https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

        """        
        nDims = len(coordMeshTuple)
        if nDims < 2:
            raise TypeError("Expected nDims >= 2; recieved "+str(nDims))
            
        uniqueCoords = [np.unique(c) for c in coordMeshTuple]
        
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
        

        Parameters
        ----------
        returnAllPoints : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        allowedEndpoints : TYPE
            DESCRIPTION.
        allowedIndices : TYPE

        """
        if returnAllPoints:
            warnings.warn("find_endpoints_on_grid is finding all "\
                          +"contours; this may include starting point")
        
        nDims = len(coordMeshTuple)
        uniqueCoords = [np.unique(c) for c in coordMeshTuple]
        
        potArr = _get_correct_shape(uniqueCoords,potArr)
        
        allContours = SurfaceUtils.find_approximate_contours(coordMeshTuple,potArr,eneg=eneg)
        
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
    the array.

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

def _get_correct_shape(gridPoints,arrToCheck):
    """
    Utility for automatically correcting the shape of an array, to deal with
    nonsense regarding np.meshgrid's default setup

    Parameters
    ----------
    gridPoints : TYPE
        DESCRIPTION.
    arrToCheck : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    defaultMeshgridShape = np.array([len(g) for g in gridPoints])
    possibleOtherShape = tuple(defaultMeshgridShape)
    defaultMeshgridShape[[1,0]] = defaultMeshgridShape[[0,1]]
    defaultMeshgridShape = tuple(defaultMeshgridShape)
    
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
    
    :Maintainer: Daniel
    """
    def __init__(self,gridPoints,gridVals,boundaryHandler="exponential",symmExtend=None,\
                 transformFuncName="identity",splKWargs={}):
        """
        Initializes the class instance. Carries out basic error checking on inputs.
        Defines self._call as the method to evaluate a point that's within the
        grid boundaries. It is also used in the boundary handlers, to evaluate
        e.g. the nearest allowed point once it is found.

        Parameters
        ----------
        gridPoints : tuple of ndarrays
            The unique grid points. Each array must be sorted in ascending order.
        gridVals : ndarray
            The grid values to be interpolated. Expected to be of shape (N2,N1,N3,...),
            as in the output of np.meshgrid.
        boundaryHandler : str, optional
            How points outside of the interpolation region are handled. The 
            default is 'exponential'.
        symmExtend : bool or ndarray of bools, optional
            Whether to symmetrically extend gridVals when evaluating. See notes.
            The default is None.
        transformFuncName : string, optional
            The function to apply to the interpolated function after interpolating.
            The default is "identity", in which no post-processing is applied.
        splKWargs : dict, optional
            Extra arguments for spline interpolation, in the 2D case. The default
            is {}.

        Returns
        -------
        None.
        
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
        
        bdyHandlerFuncs = {"exponential":self._exp_boundary_handler}
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
        self.gridVals = _get_correct_shape(gridPoints,gridVals)
        
        for i, p in enumerate(gridPoints):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
        
        if self.nDims == 2:
            self.rbv = RectBivariateSpline(*gridPoints,self.gridVals.T,**splKWargs)
            self._call = self._call_2d
        else:
            self._call = self._call_nd
            
        postEvalDict = {"identity":self._identity_transform_function,
                        "smooth_abs":self._smooth_abs_transform_function}
        self.post_eval = postEvalDict[transformFuncName]

    def __call__(self,points):
        """
        Interpolation at coordinates.
        
        Parameters
        ----------
        points : ndarray
            The coordinates to sample the gridded data at. Can be more than 2D,
            as in points.shape == complexShape + (self.nDims,).
            
        Returns
        -------
        result : ndarray
            The interpolated function evaluated at points. Is of shape complexShape.
        
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
        
        #Checking if each point is acceptable, and interpolating individual points.
        result = np.zeros(points.shape[0])
        
        for (ptIter, point) in enumerate(points):
            isInBounds = np.zeros((2,self.nDims),dtype=bool)
            isInBounds[0] = (np.array([g[0] for g in self.gridPoints]) <= point)
            isInBounds[1] = (point <= np.array([g[-1] for g in self.gridPoints]))
            
            if np.count_nonzero(~isInBounds) == 0:
                result[ptIter] = self._call(point)
            else:
                result[ptIter] = self.boundaryHandler(point,isInBounds)
        
        result = self.post_eval(result)
        result = result.reshape(originalShape)
        
        return result
    
    def _call_2d(self,point):
        """
        Evaluates the RectBivariateSpline instance at a single point. Defined
        as a wrapper here so that self._call has the same calling signature
        regardless of dimension.

        Parameters
        ----------
        point : ndarray
            Of shape (2,). A single point to evaluate at.

        Returns
        -------
        ndarray of floats
            The RectBivariateSpline evaluation. Of shape (1,).

        """
        return self.rbv(point[0],point[1],grid=False)
    
    def _call_nd(self,point):
        """
        Repeated linear interpolation. For the 2D case, see e.g.
        https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean

        Parameters
        ----------
        point : ndarray
            The point to evaluate at. Of shape (self.nDims,)

        Returns
        -------
        value : float
            The linear interpolated value.
            
        Notes
        -----
        Original implementation, taken from scipy.interpolate.RegularGridInterpolator,
        handled multiple points at a time. I've trimmed things down here so that
        it only handles a single point at a time, since the loop in self.__call__
        has to check every point individually anyways.

        """
        indices, normDistances = self._find_indices(np.expand_dims(point,1))
                
        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        value = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, normDistances):
                weight *= np.where(ei == i, 1 - yi, yi).item()
            value += weight * self.gridVals[edge_indices].item()
        
        return value
    
    def _find_indices(self,points):
        """
        Finds indices of nearest gridpoint, utilizing the regularity of the grid.
        Also computes how far in each coordinate dimension every point is from
        the previous gridpoint (not the nearest), normalized such that the next 
        gridpoint (in a particular dimension) is distance 1 from the nearest gridpoint
        (called unity units). The distance is normed to make the interpolation 
        simpler.
        
        Taken from scipy.interpolate.RegularGridInterpolator.

        Parameters
        ----------
        points : Numpy array
            Array of coordinate(s) to evaluate at. Of shape (ndims,_)

        Returns
        -------
        indices : Tuple of ndarrays
            The indices of the nearest gridpoint for all points of points. Can
            be used as indices of a numpy array
        normDistances : ndarray
            The distance along each dimension to the nearest gridpoint. Of shape
            points.shape

        Example
        -------
        Returned indices of ([2,3],[1,7],[3,2]) indicates that the first point 
        has nearest grid index (2,1,3), and the second has nearest grid index 
        (3,7,2).
        
        Notes
        -----
        If the nearest edge is the outermost edge in a given coordinate, the inner 
        edge is return instead.
        
        Requires points to have first dimension equal to self.nDims so that
        this can zip points and self.gridPoints
        
        """
        indices = []
        normDistances = np.zeros(points.shape)
        
        for (coordIter,x,grid) in zip(np.arange(self.nDims),points,self.gridPoints):
            #This is why the grid must be sorted - this search is now quick. All
            #this does is find the index in which to place x such that the list
            #self.grid[coordIter] remains sorted.
            i = np.searchsorted(grid, x) - 1
            
            #If x would be the new first element, index it as zero
            i[i < 0] = 0
            #If x would be the last element, make it not so. However, the way
            #the interpolation scheme is set up, we need the nearest gridpoint
            #that is not the outermost gridpoint. See below with grid[i+1]
            i[i > grid.size - 2] = grid.size - 2
            
            indices.append(i)
            normDistances[coordIter] = (x - grid[i]) / (grid[i + 1] - grid[i])
            
        return tuple(indices), normDistances
    
    def _exp_boundary_handler(self,point,isInBounds):
        """
        Given a point that's out of the grid region, computes the nearest point
        in the region, evaluates there, and multiplies the result by an exponential
        scaling factor. This should smoothly continue the surface, in an effort
        to push force-based solvers back into the interpolated region.

        Parameters
        ----------
        point : ndarray
            The point that's out of bounds. Of shape (self.nDims,).
        isInBounds : ndarray of bools
            Array detailing where the point fails. Of shape (2,self.nDims).
            isInBounds[0] deals with the lower bound; isInBounds[1] deals with
            the upper bound. False means the point is out of bounds in that dimension,
            in that position.
            

        Returns
        -------
        result : float
            The scaled result.
            
        Notes
        -----
        Does not allow for evaluation of multiple points at a time.
        
        For the i'th coordinate, both isInBounds[0,i] and isInBounds[1,i] should
        not be False - a point cannot be less than a minimum value and greater
        than a maximum value.

        """
        nearestAllowed = np.zeros(point.shape)
        
        for dimIter in range(point.size):
            if np.all(isInBounds[:,dimIter]):
                nearestAllowed[dimIter] = point[dimIter]
            else:
                #To convert from tuple -> numpy array -> int
                failedInd = np.nonzero(isInBounds[:,dimIter]==False)[0].item()
                if failedInd == 1:
                    failedInd = -1
                nearestAllowed[dimIter] = self.gridPoints[dimIter][failedInd]
        
        #Evaluating the nearest allowed point on the boundary of the allowed region
        valAtNearest = self._call(point)
        
        dist = np.linalg.norm(nearestAllowed-point)
        
        #Yes, I mean to take an additional square root here
        result = valAtNearest*np.exp(np.sqrt(dist))
        return result
    
    def _identity_transform_function(self,normalEvaluation):
        """
        Not sure if it's faster to have this dummy function in place, or to have
        an "if-else" statement every time we check if we should call a transform
        function.

        Parameters
        ----------
        normalEvaluation : TYPE
            DESCRIPTION.

        Returns
        -------
        normalEvaluation : TYPE
            DESCRIPTION.

        """
        return normalEvaluation
    
    def _smooth_abs_transform_function(self,normalEvaluation):
        return np.sqrt(normalEvaluation**2 + 10**(-4))
    
class PositiveSemidefInterpolator:
    def __init__(self,gridPoints,listOfVals,ndInterpKWargs={}):
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
        ndInterpKWargs : TYPE, optional
            DESCRIPTION. The default is {}.

        Returns
        -------
        None.

        """
        self.nDims = len(gridPoints)
        self.gridPoints = gridPoints
        
        #Stupid case for nDims == 1. For higher dimensions, pass through
        #NDInterpWithBoundary in components individually
        if self.nDims != 2:
            raise NotImplementedError
        
        #Standard error checking
        assert len(listOfVals) == int(self.nDims*(self.nDims+1)/2)
        
        for i, p in enumerate(gridPoints):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
        
        self.gridValsList = [_get_correct_shape(gridPoints,l) for l in listOfVals]
        
        #Taking shortcuts because I only care about D=2 right now
        self.gridVals = np.stack((np.stack((self.gridValsList[0],self.gridValsList[1])),\
                                  np.stack((self.gridValsList[1],self.gridValsList[2]))))
        self.gridVals = np.moveaxis(self.gridVals,[0,1],[2,3])
        
        self.eigenVals, self.eigenVecs = np.linalg.eig(self.gridVals)
        thetaVals = np.arccos(self.eigenVecs[:,:,0,0])
        
        #Constructing interpolators
        self.eigenValInterps = [NDInterpWithBoundary(self.gridPoints,e,**ndInterpKWargs)\
                                for e in self.eigenVals.T]
        self.eigenVecInterp = NDInterpWithBoundary(self.gridPoints,thetaVals,**ndInterpKWargs)
        
    def __call__(self,points):
        originalShape = points.shape[:-1]
        if originalShape == ():
            originalShape = (1,)
        
        if points.shape[-1] != self.nDims:
            raise ValueError("The requested sample points have dimension "
                             "%d, but this NDInterpWithBoundary expects "
                             "dimension %d" % (points.shape[-1], self.nDims))
            
        points = points.reshape((-1,self.nDims))
        
        eigenVals = [e(points) for e in self.eigenValInterps]
        theta = self.eigenVecInterp(points)
        
        ct = np.cos(theta)
        st = np.sin(theta)
        
        eigenVals = [e.clip(0) for e in eigenVals]
        
        ret = np.zeros((len(points),2,2))
        for (ptIter,point) in enumerate(points):
            ret[ptIter,0,0] = eigenVals[0][ptIter]*ct[ptIter]**2 + \
                eigenVals[1][ptIter]*st[ptIter]**2
            ret[ptIter,1,0] = (eigenVals[1][ptIter]-eigenVals[0][ptIter])*st[ptIter]*ct[ptIter]
            ret[ptIter,0,1] = ret[ptIter,1,0]
            ret[ptIter,1,1] = eigenVals[0][ptIter]*st[ptIter]**2 + \
                eigenVals[1][ptIter]*ct[ptIter]**2
                
        return ret.reshape(originalShape+(2,2))
    
def mass_funcs_to_array_func(dictOfFuncs,uniqueKeys):
    """
    Formats a collection of functions for use in computing the inertia tensor.
    Assumes the inertia tensor is symmetric.
    
    Parameters
    ----------
    dictOfFuncs : dict
        Contains functions for each component of the inertia tensor
        
    uniqueKeys : list
        Labels the unique coordinates of the inertia tensor, in the order they
        are used in the inertia. For instance, if one uses (q20, q30) as the 
        coordinates in this order, one should feed in ['20','30'], and the
        inertia will be reshaped as
        
                    [[M_{20,20}, M_{20,30}]
                     [M_{30,20}, M_{30,30}]].
                    
        Contrast this with feeding in ['30','20'], in which the inertia will
        be reshaped as
        
                    [[M_{30,30}, M_{30,20}]
                     [M_{20,30}, M_{20,20}]].

    Returns
    -------
    func_out : function
        The inertia tensor. Can be called as func_out(coords).
        
    :Maintainer: Daniel
    """
    nDims = len(uniqueKeys)
    pairedKeys = np.array([c1+c2 for c1 in uniqueKeys for c2 in uniqueKeys]).reshape(2*(nDims,))
    dictKeys = np.zeros(pairedKeys.shape,dtype=object)
    
    for (idx, key) in np.ndenumerate(pairedKeys):
        for dictKey in dictOfFuncs.keys():
            if key in dictKey:
                dictKeys[idx] = dictKey
                
    nFilledKeys = np.count_nonzero(dictKeys)
    nExpectedFilledKeys = nDims*(nDims+1)/2
    if nFilledKeys != nExpectedFilledKeys:
        raise ValueError("Expected "+str(nExpectedFilledKeys)+" but found "+\
                         str(nFilledKeys)+" instead. dictKeys = "+str(dictKeys))
    
    def func_out(coords):
        originalShape = coords.shape[:-1]
        if originalShape == ():
            originalShape = (1,)
        
        if coords.shape[-1] != nDims:
            raise ValueError("The requested sample points have dimension "
                             "%d, but this NDInterpWithBoundary expects "
                             "dimension %d" % (coords.shape[-1], nDims))
        
        coords = coords.reshape((-1,nDims))
        
        nPoints = coords.shape[0]
        outVals = np.zeros((nPoints,)+2*(nDims,))
        
        #Mass array is always 2D
        for iIter in range(nDims):
            for jIter in np.arange(iIter,nDims):
                key = dictKeys[iIter,jIter]
                fEvals = dictOfFuncs[key](coords)
                
                outVals[:,iIter,jIter] = fEvals
                outVals[:,jIter,iIter] = fEvals
                
        return outVals.reshape(originalShape+2*(nDims,))
    return func_out

class InterpolatedPath:
    """
    :Maintainer: Daniel
    """
    def __init__(self,discretePath,kwargs={}):
        """
        Note that when one considers a 1D curve embedded in 2D, e.g. in a plot 
        of a function, one should specify 'u' in kwargs. Otherwise, 'u' will
        be computed based on the distance between points on the path, which
        will generally lead to a different plot than what is desired.

        Parameters
        ----------
        discretePath : TYPE
            DESCRIPTION.
        kwargs : TYPE, optional
            DESCRIPTION. The default is {}.

        Returns
        -------
        None.

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
        if np.any(t>1) or np.any(t<0):
            raise ValueError("t must be between 0 and 1 for interpolation")
        
        return splev(t,self.tck)
    
    def compute_along_path(self,target_func,nImages,tfArgs=[],tfKWargs={}):
        t = np.linspace(0,1,nImages)
        path = np.array(self.__call__(t)).T
        tfOut = target_func(path,*tfArgs,**tfKWargs)
        
        return path, tfOut
    
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
    maxima_pnts = SurfaceUtils.find_all_local_maximum(EnergyOnPath)[0]
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