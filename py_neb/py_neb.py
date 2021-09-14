import numpy as np
from scipy.ndimage import filters, morphology #For minimum finding

#For ND interpolation
from scipy.interpolate import interpnd, RectBivariateSpline
import itertools

import h5py
import pandas as pd
import sys
import warnings

"""
CONVENTIONS:
    -Paths should be of shape (nPoints, nDimensions)
    -Functions (e.g. a potential) that take in a single point should assume the
        first index of the array iterates over the points
    -Similarly, functions (e.g. the action) that take in many points should also
        assume the first index iterates over the points
"""

"""
Other functions we want:
    -Class for least action path and minimum energy path
    -Method(s) for finding starting points for a D-dimensional grid.
        -Would want this to be robust. Ideally, it would select the outer turning
            (D-1)-dimensional hypersurface. Simple idea is just to select *any* point
            with the same energy as the ground state, and see what happens. This brings
            up another point: is there a way to show we're at the outer turning line,
            without looking at the surface?
"""

"""
TODO:
    -Unclear how we want to handle errors (currently just using sys.exit)

"""

global fdTol
fdTol = 10**(-8)

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

def midpoint_grad(func,points,eps=10**(-8)):
    """
    Midpoint finite difference. Probably best if not used with actual DFT calculations,
        vs a forwards/reverse finite difference
    Assumes func only depends on a single point (vs the action, which depends on
         all of the points)
    """
    if len(points.shape) == 1:
        points = points.reshape((1,-1))
    nPoints, nDims = points.shape
    
    gradOut = np.zeros((nPoints,nDims))
    for ptIter in range(nPoints):
        for dimIter in range(nDims):
            step = np.zeros(nDims)
            step[dimIter] = 1
            
            forwardStep = points[ptIter] + eps/2*step
            backwardStep = points[ptIter] - eps/2*step
            
            forwardEval = func(forwardStep)
            backwardEval = func(backwardStep)
            
            gradOut[ptIter,dimIter] = (forwardEval - backwardEval)/eps
    
    return gradOut

def action(path,potential,masses=None):
    """
    Allowed masses:
        -Constant mass; set masses = None
        -Array of values; set masses to a numpy array of shape (nPoints, nDims, nDims)
        -A function; set masses to a function
    Allowed potential:
        -Array of values; set potential to a numpy array of shape (nPoints,)
        -A function; set masses to a function
        
    Computes action as
        $ S = \sum_{i=1}^{nPoints} \sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
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
    #TODO: if potArr < 0, return an error flag that tells whatever is calling
    #this to modify the potential
    if np.any(potArr<0):
        print("Path: ")
        print(path)
        print("Potential: ")
        print(potArr)
        sys.exit("Stopping")
    #Actual calculation
    actOut = 0
    for ptIter in range(1,nPoints):
        coordDiff = path[ptIter] - path[ptIter - 1]
        dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
        actOut += np.sqrt(2*potArr[ptIter]*dist)
    
    return actOut, potArr, massArr

def forward_action_grad(path,potential,potentialOnPath,mass,massOnPath,\
                        target_func):
    """
    potential and mass are as allowed in "action" func; will let that do the error
    checking (for now...?)
    
    Takes forwards finite difference approx of any action-like function
    
    Does not return the gradient of the mass function, as that's not used elsewhere
    in the algorithm
    
    Maybe put this + action inside of LeastActionPath? not sure how we want to structure that part
    """
    eps = fdTol#10**(-8)
    
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
        if len(coords.shape) == 1:
            coords = coords.reshape((1,nDims))
        elif len(coords.shape) > 2:
            raise ValueError("coords.shape = "+str(coords.shape)+\
                             "; coords.shape must have length <= 2")
        
        nPoints = coords.shape[0]
        outVals = np.zeros((nPoints,)+2*(nDims,))
        
        #Mass array is always 2D
        for iIter in range(nDims):
            for jIter in np.arange(iIter,nDims):
                key = dictKeys[iIter,jIter]
                fEvals = dictOfFuncs[key](coords)
                
                outVals[:,iIter,jIter] = fEvals
                outVals[:,jIter,iIter] = fEvals
                
        return outVals
    return func_out

def auxiliary_potential(func_in,shift=10**(-4)):
    def func_out(coords):
        return func_in(coords) + shift
    return func_out

class RectBivariateSplineWrapper(RectBivariateSpline):
    def __init__(self,*args,**kwargs):
        super(RectBivariateSplineWrapper,self).__init__(*args,**kwargs)
        self.function = self.func_wrapper()
        
    def func_wrapper(self):
        def func_out(coords):
            if coords.shape == (2,):
                coords = coords.reshape((1,2))
                
            res = self.__call__(coords[:,0],coords[:,1],grid=False)
            return res
        return func_out

class NDInterpWithBoundary:
    """
    Based on scipy.interpolate.RegularGridInterpolator
    """
    def __init__(self, points, values, boundaryHandler="exponential",minVal=0):
        if len(points) < 3:
            warnings.warn("Using ND linear interpolator with "+str(len(points))\
                          +" dimensions. Consider using spline interpolator instead.")
        
        if boundaryHandler not in ["exponential"]:
            raise ValueError("boundaryHandler '%s' is not defined" % boundaryHandler)
        
        if not hasattr(values, 'ndim'):
            #In case "values" is not an array
            values = np.asarray(values)
            
        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))
            
        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)
                
        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self.boundaryHandler = boundaryHandler
        self.minVal = minVal #To be used later, perhaps

    def __call__(self, xi):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        """
        ndim = len(self.grid)
        
        #Don't really understand what this does
        xi = interpnd._ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                              "%d, but this RegularGridInterpolator has "
                              "dimension %d" % (xi.shape[1], ndim))
        
        #Checking if each point is acceptable, and interpolating individual points.
        #Might as well just make the user deal with reshaping, unless I find I need
        #to do so repeatedly
        nPoints = int(xi.size/len(self.grid))
        result = np.zeros(nPoints)
        
        for (ptIter, point) in enumerate(xi):
            isInBounds = np.zeros((2,ndim),dtype=bool)
            isInBounds[0] = (np.array([g[0] for g in self.grid]) <= point)
            isInBounds[1] = (point <= np.array([g[-1] for g in self.grid]))
            
            if np.count_nonzero(~isInBounds) == 0:
                indices, normDistances = self._find_indices(np.expand_dims(point,1))
                result[ptIter] = self._evaluate_linear(indices, normDistances)
            else:
                if self.boundaryHandler == "exponential":
                    result[ptIter] = self._exp_boundary_handler(point,isInBounds)
                
        return result
    
    def _find_indices(self, xi):
        """
        Finds indices of nearest gridpoint (utilizing the regularity of the grid).
        Also computes how far in each coordinate dimension every point is from
        the previous gridpoint (not the nearest), normalized such that the next 
        gridpoint (in a particular dimension) is distance 1 from the nearest gridpoint.
        The distance is normed to make the interpolation simpler.
        
        As an example, returned indices of [[2,3],[1,7],[3,2]] indicates that the
        first point has nearest grid index (2,1,3), and the second has nearest
        grid index (3,7,2).
        
        Note that, if the nearest edge is the outermost edge in a given coordinate,
        the inner edge is return instead.

        Parameters
        ----------
        xi : Numpy array
            Array of coordinate(s) to evaluate at. Of dimension (ndims,_)

        Returns
        -------
        indices : Tuple of numpy arrays
            The indices of the nearest gridpoint for all points of xi. Can
            be used as indices of a numpy array
        normDistances : List of numpy arrays
            The distance along each dimension to the nearest gridpoint

        """
        
        indices = []
        # compute distance to lower edge in unity units
        normDistances = []
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
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
            normDistances.append((x - grid[i]) / (grid[i + 1] - grid[i]))
            
        return tuple(indices), normDistances

    def _evaluate_linear(self, indices, normDistances):
        """
        A complicated way of implementing repeated linear interpolation. See
        e.g. https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean
        for the 2D case. Note that the weights simplify because of the normed
        distance that's returned from self._find_indices

        Parameters
        ----------
        indices : TYPE
            DESCRIPTION.
        normDistances : TYPE
            DESCRIPTION.

        Returns
        -------
        values : TYPE
            DESCRIPTION.

        """
        #TODO: remove extra dimension handling
        # slice for broadcasting over trailing dimensions in self.values.
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))
        
        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, normDistances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values
    
    def _exp_boundary_handler(self,point,isInBounds):
        nearestAllowed = np.zeros(point.shape)
        
        for dimIter in range(point.size):
            if np.all(isInBounds[:,dimIter]):
                nearestAllowed[dimIter] = point[dimIter]
            else:
                #To convert from tuple -> numpy array -> int
                failedInd = np.nonzero(isInBounds[:,dimIter]==False)[0].item()
                if failedInd == 1:
                    failedInd = -1
                nearestAllowed[dimIter] = self.grid[dimIter][failedInd]
        
        #Evaluating the nearest allowed point on the boundary of the allowed region
        indices, normDist = self._find_indices(np.expand_dims(nearestAllowed,1))
        valAtNearest = self._evaluate_linear(indices,normDist)
        
        dist = np.linalg.norm(nearestAllowed-point)
        
        #Yes, I mean to take an additional square root here
        result = valAtNearest*np.exp(np.sqrt(dist))
        return result

class LeastActionPath:
    """
    class documentation...?
    """
    def __init__(self,potential,nPts,nDims,mass=None,endpointSpringForce=True,\
                 endpointHarmonicForce=True,target_func=action,\
                 target_func_grad=forward_action_grad,nebParams={}):
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
        
        for key in nebParams.keys():
            setattr(self,key,nebParams[key])
            
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
        
        return netForce

#TODO: maybe rename something like ForceMinimization?   
#TODO: do we compute the action for all points after optimizing? or let someone else do that? 
class VerletMinimization:
    def __init__(self,nebObj,initialPoints):
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
            {"dtMax":0.1,"nAccel":5,"fInc":1.1,"fAccel":0.99,"fDecel":0.5,"aStart":0.1}
            
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
            #TODO: move to _local and _global iters (multiplication doesn't work -_-)
            self.allPts[step] = self.allPts[step-1] + \
                tStepArr[step-1]*self.allVelocities[step-1] + \
                0.5*self.allForces[step-1]*tStepArr[step-1]**2
            self.allForces[step] = self.nebObj.compute_force(self.allPts[step])
            #Warning to user: velocity here is not final velocity
            self.allVelocities[step] = \
                0.5*tStepArr[step-1]*(self.allForces[step]+self.allForces[step-1])
            
            if useLocal:
                tStepArr,alphaArr,stepsSinceReset = \
                    self._local_fire_iter(step,tStepArr,alphaArr,stepsSinceReset,\
                                          fireParams)
            else:
                tStepArr,alphaArr,stepsSinceReset = \
                    self._global_fire_iter(step,tStepArr,alphaArr,stepsSinceReset,\
                                           fireParams)
                        
        self.allPts[-1] = self.allPts[-2] + tStepArr[-1]*self.allVelocities[-1] + \
            0.5*self.allForces[-1]*tStepArr[-1]**2
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _local_fire_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        for ptIter in range(self.nPts):
            alpha = alphaArr[step-1,ptIter]
            
            product = np.dot(self.allVelocities[step,ptIter],self.allForces[step,ptIter])
            if product > 0:
                vMag = np.linalg.norm(self.allVelocities[step,ptIter])
                fHat = self.allForces[step,ptIter]/np.linalg.norm(self.allForces[step,ptIter])
                self.allVelocities[step,ptIter] = (1-alpha)*self.allVelocities[step,ptIter] + \
                    alpha*vMag*fHat
                
                if stepsSinceReset[ptIter] > fireParams["nAccel"]:
                    tStepArr[step,ptIter] = \
                        min(tStepArr[step-1,ptIter]*fireParams["fInc"],fireParams["dtMax"])
                    alphaArr[step,ptIter] = alpha*fireParams["fAccel"]
                
                stepsSinceReset[ptIter] += 1
            else:
                tStepArr[step,ptIter] = tStepArr[step-1,ptIter]*fireParams["fDecel"]
                self.allVelocities[step,ptIter] = np.zeros(self.nDims)
                alphaArr[step,ptIter] = fireParams["aStart"]
                stepsSinceReset[ptIter] = 0
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _global_fire_iter(self,step,tStepArr,alphaArr,stepsSinceReset,fireParams):
        for ptIter in range(self.nPts):
            alpha = alphaArr[step-1]
            
            product = np.dot(self.allVelocities[step,ptIter],self.allForces[step,ptIter])
            if product > 0:
                vMag = np.linalg.norm(self.allVelocities[step,ptIter])
                fHat = self.allForces[step,ptIter]/np.linalg.norm(self.allForces[step,ptIter])
                self.allVelocities[step,ptIter] = (1-alpha)*self.allVelocities[step,ptIter] \
                    + alpha*vMag*fHat
                
                if stepsSinceReset > fireParams["nAccel"]:
                    tStepArr[step] = min(tStepArr[step-1]*fireParams["fInc"],fireParams["dtMax"])
                    alphaArr[step] = alpha*fireParams["fAccel"]
                
                stepsSinceReset += 1
            else:
                tStepArr[step] = tStepArr[step-1]*fireParams["fDecel"]
                self.allVelocities[step,ptIter] = np.zeros(self.nDims)
                alphaArr[step] = fireParams["aStart"]
                stepsSinceReset = 0
        
        return tStepArr, alphaArr, stepsSinceReset
    
    def _check_early_stop(self):
        
        return None
    



    