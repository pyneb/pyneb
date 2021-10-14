from fileio import *

#Appears to be common/best practice to import required packages in every file
#they are used in
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools

from scipy.interpolate import interpnd, RectBivariateSpline, splprep, splev
from scipy.ndimage import filters, morphology #For minimum finding
import warnings

global fdTol
fdTol = 10**(-8)

class TargetFunctions:
    #No need to do any compatibility checking with gradients here.
    @staticmethod
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
            $ S = sum_{i=1}^{nPoints} sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
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
        
        for ptIter in range(nPoints):
            if potArr[ptIter] < 0:
                potArr[ptIter] = 0.01
            
        #Actual calculation
        actOut = 0
        for ptIter in range(1,nPoints):
            coordDiff = path[ptIter] - path[ptIter - 1]
            dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
            actOut += np.sqrt(2*potArr[ptIter]*dist)
        
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

        '''
        raise DeprecationWarning("If you actually use this, please let Daniel know")
        
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
    #Should check compatibility here (at least have a list of compatible actions
    #to check in *other* methods)
    #Fill out as appropriate
    def discrete_sqr_action_grad(self,path,potential,potentialOnPath,mass,massOnPath,\
                                 target_func):
        """
        
        Performs discretized action gradient, needs numerical PES still
     
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
                (mu*potentialOnPath[ptIter] + mu*potentialOnPath[ptIter-1])*dhat-\
                (mu*potentialOnPath[ptIter] + mu*potentialOnPath[ptIter+1])*dhatP1+\
                mu*gradOfPes[ptIter]*(dnorm+dnormP1))
        
        return gradOfAction, gradOfPes
    
    def discrete_action_grad(self,path,potential,potentialOnPath,mass,massOnPath,\
                            target_func):
        """
        
        Performs discretized action gradient, needs numerical PES still
     
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
        potential and mass are as allowed in "action" func; will let that do the error
        checking (for now...?)
        
        Takes forwards finite difference approx of any action-like function
        
        Does not return the gradient of the mass function, as that's not used elsewhere
        in the algorithm
        
        Maybe put this + action inside of LeastActionPath? not sure how we want to structure that part
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

def potential_central_grad(points,potential,auxFunc=None):
    '''
    TODO: is this actually... used anywhere?

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


class SurfaceUtils:
    """
    Defined for namespace purposes
    """
    @staticmethod
    def find_all_local_minimum(arr):
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
        eroded_background = morphology.binary_erosion(background,\
                                                      structure=neighborhood,\
                                                      border_value=1)
            
        detected_minima = local_min ^ eroded_background
        allMinInds = np.vstack(local_min.nonzero())
        minIndsOut = tuple([allMinInds[coordIter,:] for \
                            coordIter in range(allMinInds.shape[0])])
        return minIndsOut
    
    def find_local_minimum(arr,searchPerc=[0.25,0.25],returnOnlySmallest=True):
        """
        Returns the indices corresponding to the local minimum values within a
        desired part of the PES.
        
        Parameters
        ----------
        arr : Numpy array
            A D-dimensional array.
        searchPerc : List
            Percentage of each coordinate that the minimum is allowed to be in
    
        Returns
        -------
        minIndsOut : Tuple of numpy arrays
            D arrays of length k, for k minima found
    
        """
        if len(searchPerc) != len(arr.shape):
            raise TypeError("searchPerc and arr have unequal lengths ("+\
                            str(len(searchPerc))+") and ("+str(len(arr.shape))+")")
        
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
        nDims = len(coordMeshTuple)
        
        fig, ax = plt.subplots()
        
        if nDims == 1:
            sys.exit("Err: weird edge case I haven't handled. Why are you looking at D=1?")
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
    def round_points_to_grid(coordMeshTuple,ptsArr,dimOrder="meshgrid"):
        """
        

        Parameters
        ----------
        coordMeshTuple : TYPE
            DESCRIPTION.
        ptsArr : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if dimOrder not in ["meshgrid","human"]:
            raise ValueError("dimOrder "+str(dimOrder)+" not recognized")
        
        nDims = len(coordMeshTuple)
        if nDims < 2: #TODO: probably useless, but could be nice for completion
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
            # Some indexing is done to deal with the default shape of np.meshgrid.
            # For D dimensions, the output is of shape (N2,N1,N3,...,ND), while the
            # way indices are generated expects a shape of (N1,...,ND). So, I swap
            # the first two indices by hand. See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
            if dimOrder == "meshgrid":
                inds[[0,1]] = inds[[1,0]]
            inds = tuple(inds)
            gridValsOut[ptIter] = np.array([c[inds] for c in coordMeshTuple])
            
        if dimOrder == "meshgrid":
            #Expect columns of returned indices to be in order (N1,N2,N3,...,ND)
            indsOut[:,[0,1]] = indsOut[:,[1,0]]
        
        return indsOut, gridValsOut
    
    @staticmethod
    def find_endpoints_on_grid(coordMeshTuple,potArr,returnAllPoints=False,eneg=0,\
                               dimOrder="meshgrid"):
        """
        

        Parameters
        ----------
        returnAllPoints : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        allowedEndpoints : TYPE
            DESCRIPTION.

        """
        if returnAllPoints:
            warnings.warn("find_endpoints_on_grid is finding all "\
                          +"contours; this may include starting point")
        
        nDims = len(coordMeshTuple)
        uniqueCoords = [np.unique(c) for c in coordMeshTuple]
        
        allContours = SurfaceUtils.find_approximate_contours(coordMeshTuple,potArr,eneg=eneg)
        
        allowedEndpoints = np.zeros((0,nDims))
        allowedIndices = np.zeros((0,nDims),dtype=int)
        
        for contOnLevel in allContours:
            gridContOnLevel = []
            gridIndsOnLevel = []
            for cont in contOnLevel:
                locGridInds, locGridVals = \
                    SurfaceUtils.round_points_to_grid(coordMeshTuple,cont,dimOrder=dimOrder)
                
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
        
        return allowedEndpoints, allowedIndices

def shift_func(func_in,shift=10**(-4)):
    """
    Shifts function by shift

    Parameters
    ----------
    func_in : function
    shift : float
        The amount to shift by. The default is 10**(-4).

    Returns
    -------
    func_out : function
        The shifted function

    """
    def func_out(coords):
        return func_in(coords) - shift
    return func_out

class RectBivariateSplineWrapper(RectBivariateSpline):
    def __init__(self,*args,**kwargs):
        warnings.warn("Deprecating RectBivariateSplineWrapper in favor of "\
                      +"2D method in NDInterpWithBoundary")
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
        the inner edge is return instead. For this reason, this is a custom method
        here, despite similar logic being used elsewhere.
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
    
class NDInterpWithBoundary_experimental:
    """
    Based on scipy.interpolate.RegularGridInterpolator
    """
    def __init__(self, gridPoints, gridVals, boundaryHandler="exponential", symmExtend=None,\
                 splKWargs={}):
        """
        

        Parameters
        ----------
        gridPoints : TYPE
            DESCRIPTION.
        gridVals : TYPE
            DESCRIPTION.
        boundaryHandler : str, optional
            How points outside of the interpolation region are handled. Is assumed
            to be the same for all dimensions, because I can't think of a reasonable
            way to allow for different handling for different dimensions. I also see
            no reason why one would want to treat the dimensions differently.
            The default is 'exponential'.
        symmExtend : TYPE, optional
            DESCRIPTION. The default is None.
        splKWargs : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.nDims = len(gridPoints)
        
        bdyHandlerFuncs = {"exponential":self._exp_boundary_handler}
        if boundaryHandler not in bdyHandlerFuncs.keys():
            raise ValueError("boundaryHandler '%s' is not defined" % b)
        
        self.boundaryHandler = bdyHandlerFuncs[boundaryHandler]
        
        if symmExtend is None:
            symmExtend = np.array([False,True]+(self.nDims-2)*[False],dtype=bool)
        elif not isinstance(symmExtend,np.ndarray):
            warnings.warn("Using symmetric extension "+str(symmExtend)+\
                          " for all dimensions. Make sure this is intended.")
            symmExtend = symmExtend * np.ones(len(points),dtype=bool)
            
        if symmExtend.shape != (self.nDims,):
            raise ValueError("symmExtend.shape '"+str(symmExtend.shape)+\
                             "' does not match nDims, "+str(self.nDims))
        
        self.symmExtend = symmExtend
        
        if self.nDims == 2:
            expectedShape = tuple([len(g) for g in gridPoints])
            if gridVals.shape == expectedShape:
                self.rbv = RectBivariateSpline(*gridPoints,gridVals,**splKWargs)
            elif gridVals.T.shape == expectedShape:
                self.rbv = RectBivariateSpline(*gridPoints,gridVals.T,**splKWargs)
            else:
                raise ValueError("gridVals.shape does not match expected shape "+\
                                 str(expectedShape))
            self._call = self._call_2d
        else:
            self._call = self._call_nd
        # if len(points) > values.ndim:
        #     raise ValueError("There are %d point arrays, but values has %d "
        #                      "dimensions" % (len(points), values.ndim))
            
        # if hasattr(values, 'dtype') and hasattr(values, 'astype'):
        #     if not np.issubdtype(values.dtype, np.inexact):
        #         values = values.astype(float)
                
        # for i, p in enumerate(points):
        #     if not np.all(np.diff(p) > 0.):
        #         raise ValueError("The points in dimension %d must be strictly "
        #                          "ascending" % i)
        #     if not np.asarray(p).ndim == 1:
        #         raise ValueError("The points in dimension %d must be "
        #                          "1-dimensional" % i)
        #     if not values.shape[i] == len(p):
        #         raise ValueError("There are %d points and %d values in "
        #                          "dimension %d" % (len(p), values.shape[i], i))
                
        self.gridPoints = tuple([np.asarray(p) for p in gridPoints])
        self.gridVals = gridVals
        # self.minVal = minVal #To be used later, perhaps

    def __call__(self, points):
        """
        Interpolation at coordinates
        Parameters
        ----------
        points : np.ndarray of shape (-, self.nDims)
            The coordinates to sample the gridded data at
        """
        originalShape = points.shape[:-1]
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
                
        return result.reshape(originalShape)
    
    def _call_2d(self,point):
        return self.rbv(point[0],point[1],grid=False)
    
    def _call_nd(self,point):
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
        indices, normDistances = self._find_indices(np.expand_dims(point,1))
        
        #TODO: remove extra dimension handling
        # slice for broadcasting over trailing dimensions in self.values.
        vslice = (slice(None),) + (None,)*(self.nDims - len(indices))
        
        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        value = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, normDistances):
                weight *= np.where(ei == i, 1 - yi, yi)
            value += np.asarray(self.gridVals[edge_indices]) * weight[vslice]
        
        return value
    
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
        the inner edge is return instead. For this reason, this is a custom method
        here, despite similar logic being used elsewhere.

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
        valAtNearest = self._call(point)
        
        dist = np.linalg.norm(nearestAllowed-point)
        
        #Yes, I mean to take an additional square root here
        result = valAtNearest*np.exp(np.sqrt(dist))
        return result
    
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

class InterpolatedPath:
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
        
        defaultKWargs = {"full_output":True,"s":0}
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

#%% Deprecated functions
def potential_target_func(points, potential, auxFunc=None):
    '''

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

    '''
    warnings.warn("potential_target_func is deprecated; use TargetFunctions.mep_default",\
                  DeprecationWarning)
    ## essentially a wrapper function for the potential
    ### expected points to be a (nPts,nDim) matrix.
    ### potential should be a function capable of returning (nPts,nDim) matrix
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
def discrete_sqr_action_grad(path,potential,potentialOnPath,mass,massOnPath,\
                        target_func):
    """
    
    Performs discretized action gradient, needs numerical PES still
 
    """
    warnings.warn("discrete_sqr_action_grad deprecated, use"+\
                  " GradientApproximations().discrete_sqr_action_grad",DeprecationWarning)
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

        mu=massOnPath[ptIter,0,0]#/hbarc**2

        gradOfAction[ptIter] = 0.5*(\
            (mu*potentialOnPath[ptIter] + mu*potentialOnPath[ptIter-1])*dhat-\
            (mu*potentialOnPath[ptIter] + mu*potentialOnPath[ptIter+1])*dhatP1+\
            mu*gradOfPes[ptIter]*(dnorm+dnormP1))
    
    return gradOfAction, gradOfPes

def discrete_action_grad(path,potential,potentialOnPath,mass,massOnPath,\
                        target_func):
    """
    
    Performs discretized action gradient, needs numerical PES still
 
    """
    warnings.warn("discrete_action_grad is deprecated; use GradientApproximations().discrete_action_grad",\
                  DeprecationWarning)
    eps = fdTol#10**(-8)
    
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

        mu=massOnPath[ptIter,0,0]#/hbarc**2
        gradOfAction[ptIter] = 0.5*(\
            (np.sqrt(2*mu*potentialOnPath[ptIter]) + np.sqrt(2*mu*potentialOnPath[ptIter-1]))*dhat-\
            (np.sqrt(2*mu*potentialOnPath[ptIter]) + np.sqrt(2*mu*potentialOnPath[ptIter+1]))*dhatP1+\
            mu*gradOfPes[ptIter]*(dnorm+dnormP1) / np.sqrt(2*mu*potentialOnPath[ptIter]))
    
    return gradOfAction, gradOfPes

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
    warnings.warn("forward_action_grad is deprecated; use GradientApproximations().forward_action_grad",\
                  DeprecationWarning)
    
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
    warnings.warn("find_local_minimum is deprecated;"\
                  +" use SurfaceUtils.find_local_minimum",DeprecationWarning)
    neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood,\
                                        mode="nearest")==arr)
    
    background = (arr==0)
    eroded_background = morphology.binary_erosion(background,\
                                                  structure=neighborhood,\
                                                  border_value=1)
        
    detected_minima = local_min ^ eroded_background
    allMinInds = np.vstack(local_min.nonzero())
    minIndsOut = tuple([allMinInds[coordIter,:] for \
                        coordIter in range(allMinInds.shape[0])])
    return minIndsOut

def find_approximate_contours(coordMeshTuple,zz,eneg=0,show=False):
    warnings.warn("find_approximate_contours is deprecated;"\
                  +" use SurfaceUtils.find_approximate_contours",DeprecationWarning)
    
    nDims = len(coordMeshTuple)
    
    fig, ax = plt.subplots()
    
    if nDims == 1:
        sys.exit("Err: weird edge case I haven't handled. Why are you looking at D=1?")
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

def round_points_to_grid(coordMeshTuple,ptsArr,dimOrder="meshgrid"):
    """
    

    Parameters
    ----------
    coordMeshTuple : TYPE
        DESCRIPTION.
    ptsArr : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    warnings.warn("round_points_to_grid is deprecated;"\
                  +" use SurfaceUtils.round_points_to_grid",DeprecationWarning)
    
    if dimOrder not in ["meshgrid","human"]:
        raise ValueError("dimOrder "+str(dimOrder)+" not recognized")
    
    nDims = len(coordMeshTuple)
    if nDims < 2: #TODO: probably useless, but could be nice for completion
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
        # Some indexing is done to deal with the default shape of np.meshgrid.
        # For D dimensions, the output is of shape (N2,N1,N3,...,ND), while the
        # way indices are generated expects a shape of (N1,...,ND). So, I swap
        # the first two indices by hand. See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        if dimOrder == "meshgrid":
            inds[[0,1]] = inds[[1,0]]
        inds = tuple(inds)
        gridValsOut[ptIter] = np.array([c[inds] for c in coordMeshTuple])
        
    if dimOrder == "meshgrid":
        #Expect columns of returned indices to be in order (N1,N2,N3,...,ND)
        indsOut[:,[0,1]] = indsOut[:,[1,0]]
    
    return indsOut, gridValsOut

def find_endpoints_on_grid(coordMeshTuple,potArr,returnAllPoints=False,eneg=0,\
                           dimOrder="meshgrid"):
    warnings.warn("find_endpoints_on_grid is deprecated;"\
                  +" use SurfaceUtils.find_endpoints_on_grid",DeprecationWarning)
    """
    

    Parameters
    ----------
    returnAllPoints : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    allowedEndpoints : TYPE
        DESCRIPTION.

    """
    if returnAllPoints:
        warnings.warn("find_endpoints_on_grid is finding all "\
                      +"contours; this may include starting point")
    
    nDims = len(coordMeshTuple)
    uniqueCoords = [np.unique(c) for c in coordMeshTuple]
    
    allContours = find_approximate_contours(coordMeshTuple,potArr,eneg=eneg)
    
    allowedEndpoints = np.zeros((0,nDims))
    allowedIndices = np.zeros((0,nDims),dtype=int)
    
    for contOnLevel in allContours:
        gridContOnLevel = []
        gridIndsOnLevel = []
        for cont in contOnLevel:
            locGridInds, locGridVals = \
                round_points_to_grid(coordMeshTuple,cont,dimOrder=dimOrder)
            
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
    
    return allowedEndpoints, allowedIndices

def auxiliary_potential(func_in,shift=10**(-4)):
    warnings.warn("auxiliary_potential is deprecated, use shift_func",DeprecationWarning)
    def func_out(coords):
        return func_in(coords) + shift
    return func_out

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
        $ S = sum_{i=1}^{nPoints} sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
    """
    warnings.warn("action is deprecated, use TargetFunctions.action",DeprecationWarning)
    
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
    
    for ptIter in range(nPoints):
        if potArr[ptIter] < 0:
            potArr[ptIter] = 0.01
    
    # if np.any(potArr[1:-2]<0):
    #     print("Path: ")
    #     print(path)
    #     print("Potential: ")
    #     print(potArr)
    #     raise ValueError("Encountered energy E < 0; stopping.")
        
    #Actual calculation
    actOut = 0
    for ptIter in range(1,nPoints):
        coordDiff = path[ptIter] - path[ptIter - 1]
        dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
        actOut += np.sqrt(2*potArr[ptIter]*dist)
    
    return actOut, potArr, massArr

def action_sqr(path,potential,masses=None):
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

    '''
    """    
    Computes action as
        $ S = sum_{i=1}^{nPoints} E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b $
    """
    warnings.warn("action_sqr is deprecated, use TargetFunctions.action_squared",DeprecationWarning)
    
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
    
    for ptIter in range(nPoints):
        if potArr[ptIter] < 0:
            potArr[ptIter] = 0.01
    
    # if np.any(potArr[1:-2]<0):
    #     print("Path: ")
    #     print(path)
    #     print("Potential: ")
    #     print(potArr)
    #     raise ValueError("Encountered energy E < 0; stopping.")
        
    #Actual calculation
    actOut = 0
    for ptIter in range(1,nPoints):
        coordDiff = path[ptIter] - path[ptIter - 1]
        dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
        actOut += potArr[ptIter]*dist
    return actOut, potArr, massArr
