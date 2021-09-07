import numpy as np
from scipy.ndimage import filters, morphology #For minimum finding

#For ND interpolation
from scipy.interpolate import interpnd
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

class GridInterpWithBoundary:
    """
    Based on scipy.interpolate.RegularGridInterpolator
    """
    def __init__(self, points, values, boundaryHandler="exponential",minVal=0):
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
        self.minVal = minVal

    def __call__(self, xi):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        """
        ndim = len(self.grid)
        xi = interpnd._ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))
            
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        
        #Iterating over dimensions and checking for out-of-bounds
        isInBounds = np.zeros((2,)+xi.T.shape,dtype=bool)
        for i, p in enumerate(xi.T):
            isInBounds[0,i] = (self.grid[i][0] <= p)
            isInBounds[1,i] = (p <= self.grid[i][-1])

        indices, norm_distances = self._find_indices(xi.T)
        
        resultAssumingInBounds = self._evaluate_linear(indices, norm_distances)
        if self.boundaryHandler == "exponential":
            result = self._exp_boundary_handler(xi,resultAssumingInBounds,isInBounds,\
                                                indices)
        
        if self.minVal is not None:
            for rIter in range(len(result)):
                if result[rIter] < self.minVal:
                    result[rIter] = self.minVal
        return result

    def _evaluate_linear(self, indices, norm_distances):
        """
        A complicated way of implementing repeated linear interpolation. See
        e.g. https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation
        for the 2D case.

        Parameters
        ----------
        indices : TYPE
            DESCRIPTION.
        norm_distances : TYPE
            DESCRIPTION.

        Returns
        -------
        values : TYPE
            DESCRIPTION.

        """
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))
        
        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values
    
    def _exp_boundary_handler(self,xi,resultIn,isInBounds,indices):
        resultOut = resultIn.copy()
        
        for ptIter in range(len(resultIn)):
            if np.count_nonzero(~isInBounds[:,:,ptIter]) > 0:
                nearestAllowed = np.zeros(len(self.grid))
                for coordIter in range(len(self.grid)):
                    if np.count_nonzero(~isInBounds[:,coordIter,ptIter]) > 0:
                        nearestAllowed[coordIter] = self.grid[coordIter][indices[coordIter][ptIter]]
                    else:
                        nearestAllowed[coordIter] = xi[ptIter,coordIter]
                
                dist = np.linalg.norm(xi[ptIter] - nearestAllowed)
                
                locInds, locDists = self._find_indices(np.expand_dims(nearestAllowed,1))
                nearestActualVal = self._evaluate_linear(locInds,locDists)[0]
                
                #Yes, I mean to take an additional square root here
                resultOut[ptIter] = nearestActualVal*np.exp(np.sqrt(dist))
        
        return resultOut

    def _find_indices(self, xi):
        """
        Finds indices of nearest gridpoint (utilizing the regularity of the grid).
        Also computes how far in each coordinate dimension every point is from
        the previous gridpoint (not the nearest), normalized such that the next 
        gridpoint (in a particular dimension) is distance 1 from the nearest gridpoint.
        
        As an example, returned indices of [[2,3],[1,7],[3,2]] indicates that the
        first point has nearest grid index (2,1,3), and the second has nearest
        grid index (3,7,2).

        Parameters
        ----------
        xi : Numpy array
            Array of coordinate(s) to evaluate at. Of dimension (ndims,_)

        Returns
        -------
        indices : Tuple of numpy arrays
            The indices of the nearest gridpoint for all points of xi. Can
            be used as indices of a numpy array
        norm_distances : List of numpy arrays
            The distance along each dimension to the nearest gridpoint

        """
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
        return tuple(indices), norm_distances

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
    
class VerletMinimization:
    def __init__(self,nebObj):
        #It'll probably do this automatically, but whatever
        if not hasattr(nebObj,"compute_force"):
            raise AttributeError("Object "+str(nebObj)+" has no attribute compute_force")
            
        self.nebObj = nebObj
    
# class Minimum_energy_path_NEB():


    