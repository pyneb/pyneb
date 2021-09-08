import sys
import os
sys.path.append(os.path.expanduser("~/Research/ActionMinimization/"))

from py_neb.py_neb import *
from py_neb import py_neb

import matplotlib.pyplot as plt

import time
import datetime
import cProfile
import itertools
 
#TODO: namespace stuff here
from scipy.interpolate import interp2d, RBFInterpolator, interpn
from scipy import interpolate
import scipy.interpolate
import sklearn.gaussian_process as gp

class Utilities:
    @staticmethod
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
            D arrays of length k, for k minima found. Each tuple is all of the minima
            for a coordinate. So, e.g. ([2,3],[0,12],[7,48]) found 2 minima in
            a 3-dimensional array: one at the indices (2,0,7), and another at
            the indices (3,12,48). Returned in this format so that one can call
            xx[minIndsOut] to get the x coordinates of all minima, where xx is a
            meshgrid of x coordinates (an output of np.meshgrid(*args)).
    
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
    
    @staticmethod
    def extract_gs_inds(allMinInds,coordMeshTuple,zz,pesPerc=0.5):
        #Uses existing indices, in case there's some additional filtering I need to
        #do after calling "find_local_minimum"
        if not isinstance(pesPerc,np.ndarray):
            pesPerc = np.array(len(coordMeshTuple)*[pesPerc])
            
        nPts = zz.shape
        maxInd = np.array(nPts)*pesPerc
        
        allowedIndsOfIndices = np.ones(len(allMinInds[0]),dtype=bool)
        for cIter in range(len(coordMeshTuple)):
            allowedIndsOfIndices = np.logical_and(allowedIndsOfIndices,allMinInds[cIter]<maxInd[cIter])
            
        allowedMinInds = tuple([inds[allowedIndsOfIndices] for inds in allMinInds])
        actualMinIndOfInds = np.argmin(zz[allowedMinInds])
        
        gsInds = tuple([inds[actualMinIndOfInds] for inds in allowedMinInds])
        
        return gsInds
    
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
    def find_base_dir():
        #Auto-config for my desktop or the HPCC
        possibleHomeDirs = ["~/Research/ActionMinimization/","~/Documents/ActionMinimization/"]
        
        baseDir = None
        for d in possibleHomeDirs:
            if os.path.isdir(os.path.expanduser(d)):
                baseDir = os.path.expanduser(d)
        if baseDir is None:
            sys.exit("Err: base directory not found")
            
        return baseDir
    
    @staticmethod
    def h5_get_keys(obj):
        #Taken from https://stackoverflow.com/a/59898173
        keys = (obj.name,)
        if isinstance(obj,h5py.Group):
            for key, value in obj.items():
                if isinstance(value,h5py.Group):
                    #Is recursive here
                    keys = keys + Utilities.h5_get_keys(value)
                else:
                    keys = keys + (value.name,)
        return keys
    
    # @staticmethod
    # def initial_on_contour(coordMeshTuple,zz,nPts,debug=False):
    #     """
    #     Connects a straight line between the metastable state and the contour at
    #     the same energy. WARNING: only useful for leps_plus_ho
    #     """
    #     nCoords = len(coordMeshTuple)
        
    #     minInds = Utilities.find_local_minimum(zz)
    #     startInds = tuple([minInds[cIter][np.argmax(zz[minInds])] for\
    #                        cIter in range(nCoords)])
    #     metaEneg = zz[startInds]
        
    #     #Pulled from PES code. Doesn't generalize to higher dimensions, but not
    #     #an important issue rn.
    #     fig, ax = plt.subplots()
    #     ax.contourf(coordMeshTuple[0],coordMeshTuple[1],zz)
    #     contour = ax.contour(coordMeshTuple[0],coordMeshTuple[1],zz,\
    #                          levels=[metaEneg]).allsegs[0][0].T#Selects the actual curve
    #     if not debug:
    #         plt.close(fig)
            
    #     startPt = np.array([coordMeshTuple[tupInd][startInds] for tupInd in \
    #                         range(nCoords)])
    #     line = geometry.LineString(contour.T)
        
    #     approxFinalPt = np.array([1.5,1])
    #     point = geometry.Point(*approxFinalPt)
    #     finalPt = np.array(line.interpolate(line.project(point)))
    #     # print(point.distance(line))
            
    #     initialPoints = np.array([np.linspace(startPt[cInd],finalPt[cInd],num=nPts) for\
    #                               cInd in range(nCoords)])
        
    #     return initialPoints
    
    @staticmethod
    def new_init_on_contour(contour,gsLoc,nPts=22):
        nPoints, nCoords = contour.shape
        assert len(gsLoc) == nCoords
        
        cartesianDist = np.zeros(nPoints)
        for ptIter in range(nPoints):
            diffVec = contour[ptIter] - gsLoc
            cartesianDist[ptIter] = np.linalg.norm(diffVec)
        
        minIter = np.argmin(cartesianDist)
        initPath = np.array([np.linspace(gsLoc[cIter],contour[minIter,cIter],num=nPts) for\
                             cIter in range(nCoords)])
        
        return initPath
    
    @staticmethod
    def aux_pot(eneg_func,eGS,tol=10**(-2)):
        def pot_out(coords):
            return eneg_func(coords) - eGS + tol
        
        return pot_out
    
    @staticmethod
    def const_mass():
        def dummy_mass(coords):
            if coords[0].shape == ():
                nPoints = 1
            else:
                nPoints = len(coords[0])
            nDims = len(coords)
            retVal = np.full((nPoints,nDims,nDims),np.identity(nDims))
            return retVal
        
        return dummy_mass
    
    # @staticmethod
    # def interpolated_action(eneg_func,mass_func,discretePath,ndims=2,nPts=200):
    #     if ndims not in discretePath.shape:
    #         sys.exit("Err: path shape is "+str(discretePath.shape)+"; expected one dimension to be "+str(ndims))
            
    #     if discretePath.shape[1] == ndims:
    #         discretePath = discretePath.T
            
    #     #s=0 means that the interpolation actually passes through all of the points
    #     tck, u = interpolate.splprep(discretePath,s=0)
    #     uDense = np.linspace(0,1,num=nPts)
        
    #     pathOut = interpolate.splev(uDense,tck)
        
    #     enegs = eneg_func(*pathOut)
    #     masses = mass_func(*pathOut)
        
    #     actOut = 0
    #     for ptIter in range(1,nPts):
    #         dist2 = 0
    #         for coordIter in range(ndims):
    #             dist2 += (pathOut[coordIter][ptIter] - pathOut[coordIter][ptIter-1])**2
    #         actOut += (np.sqrt(2*masses[ptIter]*enegs[ptIter])+\
    #                    np.sqrt(2*masses[ptIter-1]*enegs[ptIter-1]))*np.sqrt(dist2)
    #     actOut = actOut/2
        
    #     return pathOut, actOut
    
    @staticmethod
    def standard_pes(xx,yy,zz,clipRange=(-5,30)):
        #TODO: pull some (cleaner) options from ML_Funcs_Class
        #Obviously not general - good luck plotting a 3D PES lol
        fig, ax = plt.subplots()
        if clipRange is None:
            clipRange = (zz.min()-0.2,zz.max()+0.2)
        #USE THIS COLORMAP FOR PESs - has minima in blue and maxima in red
        cf = ax.contourf(xx,yy,zz.clip(clipRange[0],clipRange[1]),\
                         cmap="Spectral_r",levels=np.linspace(clipRange[0],clipRange[1],25))
        plt.colorbar(cf,ax=ax)
        
        ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)")
        return fig, ax
    
def interpnd_wrapper(uniqueGridPts,gridValues,tol=10**(-5)):
    #Really, a wrapper that also puts an exponential growth once you leave the
    #interpolated region
    nCoords = len(uniqueGridPts)
    endPoints = np.array([[g[0],g[-1]] for g in uniqueGridPts])
    
    def func(coords):
        if len(coords.shape) == 1:
            coords = coords.reshape((nCoords,1))
            
        if coords.shape[0] != nCoords:
            if coords.shape[1] == nCoords:
                warnings.warn("Transposing coords; coords.shape[0] != nCoords")
                coords = coords.T
            else:
                raise ValueError("coords.shape "+str(coords.shape)+\
                                 " does not match nCoords "+\
                                 str(nCoords))
        
        nPoints = coords.shape[1]

        retVal = np.zeros(nPoints)
        for ptIter in range(nPoints):
            if np.count_nonzero(np.isnan(coords[:,ptIter])) != 0:
                sys.exit("Err: point reached with some number of NaNs")
            isAllowed = np.ones((nCoords,2),dtype=bool)
            for coordIter in range(nCoords):
                if coords[coordIter,ptIter] < endPoints[coordIter,0]:
                    isAllowed[coordIter,0] = False
                if coords[coordIter,ptIter] > endPoints[coordIter,1]:
                    isAllowed[coordIter,1] = False
            if isAllowed.all():
                retVal[ptIter] = \
                    interpn(uniqueGridPts,gridValues,coords[:,ptIter].flatten()) + tol
            else:
                # print("Path out of bounds:")
                # print(coords[ptIter])
                # print("Bad coordinate(s):")
                # print(isAllowed)
                # print(50*"=")
                #Basically projects onto the nearest allowed endpoint of the grid,
                #allowing for multiple points being off of the grid
                nearestAllowed = np.zeros(nCoords)
                for coordIter in range(nCoords):
                    if isAllowed[coordIter].all():
                        nearestAllowed[coordIter] = coords[coordIter,ptIter]
                    else:
                        failedInd = np.nonzero(isAllowed[coordIter]==False)
                        nearestAllowed[coordIter] = endPoints[coordIter,failedInd]
                dist = np.linalg.norm(nearestAllowed - coords[:,ptIter])
                #For getting the scale correct
                # try:
                evalAtNearest = interpn(uniqueGridPts,gridValues,nearestAllowed) + tol
                # except ValueError:
                #     print(nearestAllowed)
                #     print(endPoints)
                retVal[ptIter] = evalAtNearest*np.exp(np.sqrt(dist))#dist**2
        
        return retVal
    return func

# class GridInterpWithBoundary:
#     """
#     Based on scipy.interpolate.RegularGridInterpolator
#     """
#     def __init__(self, points, values, boundaryHandler="exponential",minVal=0):
#         if boundaryHandler not in ["exponential"]:
#             raise ValueError("boundaryHandler '%s' is not defined" % boundaryHandler)
        
#         if not hasattr(values, 'ndim'):
#             #In case "values" is not an array
#             values = np.asarray(values)

#         if len(points) > values.ndim:
#             raise ValueError("There are %d point arrays, but values has %d "
#                              "dimensions" % (len(points), values.ndim))

#         if hasattr(values, 'dtype') and hasattr(values, 'astype'):
#             if not np.issubdtype(values.dtype, np.inexact):
#                 values = values.astype(float)

#         for i, p in enumerate(points):
#             if not np.all(np.diff(p) > 0.):
#                 raise ValueError("The points in dimension %d must be strictly "
#                                  "ascending" % i)
#             if not np.asarray(p).ndim == 1:
#                 raise ValueError("The points in dimension %d must be "
#                                  "1-dimensional" % i)
#             if not values.shape[i] == len(p):
#                 raise ValueError("There are %d points and %d values in "
#                                  "dimension %d" % (len(p), values.shape[i], i))
        
#         self.grid = tuple([np.asarray(p) for p in points])
#         self.values = values
#         self.boundaryHandler = boundaryHandler
#         self.minVal = minVal

#     def __call__(self, xi):
#         """
#         Interpolation at coordinates
#         Parameters
#         ----------
#         xi : ndarray of shape (..., ndim)
#             The coordinates to sample the gridded data at
#         """
#         ndim = len(self.grid)
#         xi = scipy.interpolate.interpnd._ndim_coords_from_arrays(xi, ndim=ndim)
#         if xi.shape[-1] != len(self.grid):
#             raise ValueError("The requested sample points xi have dimension "
#                              "%d, but this RegularGridInterpolator has "
#                              "dimension %d" % (xi.shape[1], ndim))

#         xi_shape = xi.shape
#         xi = xi.reshape(-1, xi_shape[-1])
        
#         #Iterating over dimensions and checking for out-of-bounds
#         isInBounds = np.zeros((2,)+xi.T.shape,dtype=bool)
#         for i, p in enumerate(xi.T):
#             isInBounds[0,i] = (self.grid[i][0] <= p)
#             isInBounds[1,i] = (p <= self.grid[i][-1])

#         indices, norm_distances = self._find_indices(xi.T)
        
#         resultAssumingInBounds = self._evaluate_linear(indices, norm_distances)
#         if self.boundaryHandler == "exponential":
#             result = self._exp_boundary_handler(resultAssumingInBounds,isInBounds,\
#                                                 norm_distances,indices)
        
#         if self.minVal is not None:
#             for rIter in range(len(result)):
#                 if result[rIter] < self.minVal:
#                     result[rIter] = self.minVal
#         return result

#     def _evaluate_linear(self, indices, norm_distances):
#         # slice for broadcasting over trailing dimensions in self.values
#         vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))
        
#         # find relevant values
#         # each i and i+1 represents a edge
#         edges = itertools.product(*[[i, i + 1] for i in indices])
#         values = 0.
#         for edge_indices in edges:
#             weight = 1.
#             for ei, i, yi in zip(edge_indices, indices, norm_distances):
#                 weight *= np.where(ei == i, 1 - yi, yi)
#             values += np.asarray(self.values[edge_indices]) * weight[vslice]
#         return values
    
#     def _exp_boundary_handler(self,resultIn,isInBounds,norm_distances,indices):
#         resultOut = resultIn.copy()
        
#         gridValues = self.values[tuple(indices)]
        
#         for ptIter in range(len(resultIn)):
#             if np.count_nonzero(~isInBounds[:,:,ptIter]) > 0:
#                 dist = 0
#                 for n in norm_distances:
#                     dist += n[ptIter]**2
#                 dist = np.sqrt(dist)
#                 nearestActualVal = gridValues[ptIter]
                
#                 #Yes, I mean to take an additional square root here
#                 resultOut[ptIter] = nearestActualVal*np.exp(np.sqrt(dist))
        
#         return resultOut

#     def _find_indices(self, xi):
#         # find relevant edges between which xi are situated
#         indices = []
#         # compute distance to lower edge in unity units
#         norm_distances = []
#         # iterate through dimensions
#         for x, grid in zip(xi, self.grid):
#             i = np.searchsorted(grid, x) - 1
#             i[i < 0] = 0
#             i[i > grid.size - 2] = grid.size - 2
#             indices.append(i)
#             norm_distances.append((x - grid[i]) /
#                                   (grid[i + 1] - grid[i]))
            
#         return indices, norm_distances

def mass_array_func(arrayOfFuncs):
    #Simple wrapper for how I call things; probably changed later. Currently set
    #up such that, if the input coords is an array of size (n1, n2), the output
    #is an array of shape (n1, n2, 3, 3). For working with the action, this shouldn't
    #be an issue if I assume the 2nd and 3rd dimensions are for the different
    #collective coordinates; it's just some additional flexibility here
    def func_out(coords):
        nCoords = len(coords)
        if coords[0].shape == ():
            outputShape = (1,)
        else:
            outputShape = coords[0].shape
        outVals = np.zeros(outputShape+2*(nCoords,))
        
        #Mass array is always 2D
        for iIter in range(nCoords):
            for jIter in np.arange(iIter,nCoords):
                fEvals = arrayOfFuncs[iIter,jIter](coords)
                #A whole mess in allowing for non-flattened coordinates
                iters1 = len(outputShape)*(slice(None),)+(iIter,jIter)
                iters2 = len(outputShape)*(slice(None),)+(jIter,iIter)
                
                outVals[iters1] = fEvals
                outVals[iters2] = fEvals
                
        return outVals
    return func_out

def new_mass_array_func(arrayOfFuncs):
    #Wrapper. Can abstract this plus other
    #wrappers to allow for different shapes
    nCoords = arrayOfFuncs.shape[0]
    if not arrayOfFuncs.shape == (nCoords,nCoords):
        raise ValueError("arrayOfFuncs is not square; has shape "+str(arrayOfFuncs.shape))
        
    def func_out(coords):
        if len(coords.shape) == 1:
            coords = coords.reshape((nCoords,1))
            
        if coords.shape[0] != nCoords:
            if coords.shape[1] == nCoords:
                warnings.warn("Transposing coords; coords.shape[0] != nCoords")
                coords = coords.T
            else:
                raise ValueError("coords.shape "+str(coords.shape)+\
                                 " does not match nCoords "+\
                                 str(nCoords))
        
        nPoints = coords.shape[1]
        
        outVals = np.zeros((nPoints,nCoords,nCoords))
        
        #Mass array is always 2D
        for iIter in range(nCoords):
            for jIter in np.arange(iIter,nCoords):
                fEvals = arrayOfFuncs[iIter,jIter](coords)
                
                outVals[:,iIter,jIter] = fEvals
                outVals[:,jIter,iIter] = fEvals
                
        return outVals
    return func_out
    
class CustomInterp2d(interp2d):
    #TODO: make it impossible to go outside of the bounds of the grid data used
        #for interpolation
    def __init__(self,*args,**kwargs):
        self.kwargs = kwargs
        super(CustomInterp2d,self).__init__(*args,**kwargs)
        self.potential = self.pot_wrapper()
            
    #Need a wrapper for use in abstracted function calls
    def pot_wrapper(self):
        def potential(coords):
            coords = coords.T
            #Is a 2d method anyways
            flattenedCoords = [coords[0].flatten(),coords[1].flatten()]
            nCoords = len(coords)
            
            enegOut = np.zeros(flattenedCoords[0].shape)
            for ptIter in range(len(enegOut)):
                enegOut[ptIter] = self.__call__(flattenedCoords[0][ptIter],flattenedCoords[1][ptIter])
            
            return enegOut.reshape(coords[0].shape)
        
        return potential
    
class LocalGPRegressor():
    #WARNING: when written this way, gridPot is actually a keyword-only argument...
    #and yet, it's required.
    def __init__(self,*uniqueGridPts,gridPot,predKWargs={},gpKWargs={}):
        if gridPot is None:
            sys.exit("Err: gridPot must be specified with kwarg")
        # super(LocalGPRegressor,self).__init__(**gpKWargs)
        self.uniqueGridPts = uniqueGridPts
        
        self.inDat = np.meshgrid(*uniqueGridPts)
        self.outDat = gridPot
        self.predKWargs = predKWargs
        self.gpKWargs = gpKWargs
        self._initialize_predkwargs()
        
        self.potential = self.pot_wrapper()
        
    def _initialize_predkwargs(self):
        gridDistance = np.zeros(len(self.uniqueGridPts))
        for (gIter,grid) in enumerate(self.uniqueGridPts):
            #Assumes evenly-spaced grid. Is not quite right, but is close enough
            gridDistance[gIter] = (grid[-1] - grid[0])/len(grid)
        
        defaultKWargs = {"nNeighbors":10,"neighborDist":gridDistance}
        for arg in defaultKWargs.keys():
            if arg not in self.predKWargs.keys():
                self.predKWargs[arg] = defaultKWargs[arg]
                
        return None
    
    def pot_wrapper(self):
        def potential(coords):
            nCoords = len(self.inDat)
            
            neighborDist = self.predKWargs["neighborDist"]
            nNeighbors = self.predKWargs["nNeighbors"]
            
            if not isinstance(nNeighbors,np.ndarray):
                nNeighbors = nNeighbors*np.ones(nCoords)
            allowedDistance = nNeighbors * neighborDist
            
            flattenedCoords = [c.flatten() for c in coords]
            nPts = len(flattenedCoords[0])
            predDat = np.zeros(nPts)
            
            #TODO: can optimize this a bit by first checking if predPts are close to
            #each other, and using the same fitted GP for them if so
            for ptIter in range(nPts):
                locGP = gp.GaussianProcessRegressor(**self.gpKWargs)
                boolList = []
                for dimIter in range(nCoords):
                    locMask = (np.abs(self.inDat[dimIter] - flattenedCoords[dimIter][ptIter]) < \
                               allowedDistance[dimIter])
                    boolList.append(locMask)
                boolMask = np.logical_and.reduce(np.array(boolList))
                usedInds = tuple(np.argwhere(boolMask==True).T)
                
                locInDat = np.array([self.inDat[cIter][usedInds] for\
                                     cIter in range(nCoords)]).T
                locOutDat = self.outDat[usedInds]
                
                locGP.fit(locInDat,locOutDat)
                predPtArr = np.array([flattenedCoords[cIter][ptIter] for\
                                      cIter in range(nCoords)]).reshape((-1,nCoords))
                predDat[ptIter] = locGP.predict(predPtArr)
                # print(self.__dict__.keys())
            
            return predDat.reshape(coords[0].shape)
        return potential
    
class LepsPot():
    def __init__(self,initialGrid=np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05)),\
                 potParams={},eGS=None,loggingLevel=None,logFile=None):
        # if (loggingLevel is not None) and (logFile is None):
        #     logFile = datetime.datetime.now().isoformat()+"_log.h5"
        
        self.params = potParams
        self._initialize_params()
        
        self.initialGrid = initialGrid
        self.zz = self.leps_plus_ho(self.initialGrid)
        minInds = Utilities.find_local_minimum(self.zz)
        #Note: only good for this case, where the potential has exactly two minima.
        #This makes it so that the metastable state is at E = 0.
        if eGS is None:
            eGS = np.max(self.zz[minInds])
        
        self.potential = Utilities.aux_pot(self.leps_plus_ho,eGS)
        
    def _initialize_params(self):
        defaultDict = {"a":0.05,"b":0.8,"c":0.05,"dab":4.746,"dbc":4.746,\
                       "dac":3.445,"r0":0.742,"alpha":1.942,"rac":3.742,"kc":0.2025,\
                       "c_ho":1.154}
        for param in defaultDict.keys():
            if param not in self.params.keys():
                self.params[param] = defaultDict[param]
            
        return None
    
    def _q(self,r,d,alpha,r0):
        return d/2*(3/2*np.exp(-2*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))
    
    def _j(self,r,d,alpha,r0):
        return d/4*(np.exp(-2*alpha*(r-r0)) - 6*np.exp(-alpha*(r-r0)))
    
    def leps_pot(self,rab,rbc):
        q = self._q
        j = self._j
                
        rac = rab + rbc
        
        vOut = q(rab,self.params["dab"],self.params["alpha"],self.params["r0"])/(1+self.params["a"]) +\
            q(rbc,self.params["dbc"],self.params["alpha"],self.params["r0"])/(1+self.params["b"]) +\
            q(rac,self.params["dac"],self.params["alpha"],self.params["r0"])/(1+self.params["c"])
        
        jab = j(rab,self.params["dab"],self.params["alpha"],self.params["r0"])
        jbc = j(rbc,self.params["dbc"],self.params["alpha"],self.params["r0"])
        jac = j(rac,self.params["dac"],self.params["alpha"],self.params["r0"])
        
        jTerm = jab**2/(1+self.params["a"])**2+jbc**2/(1+self.params["b"])**2+\
            jac**2/(1+self.params["c"])**2
        jTerm = jTerm - jab*jbc/((1+self.params["a"])*(1+self.params["b"])) -\
            jbc*jac/((1+self.params["b"])*(1+self.params["c"])) -\
            jab*jac/((1+self.params["a"])*(1+self.params["c"]))
        
        vOut = vOut - np.sqrt(jTerm)
        
        return vOut
    
    def leps_plus_ho(self,rab,x):
        """
        LEPs potential plus harmonic oscillator. TODO: add source
        
        Call this function with a numpy array of rab and x:
        
            xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
            zz = leps_plus_ho(xx,yy),
        
        and plot it as
        
            fig, ax = plt.subplots()
            ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")
    
        """
        vOut = self.leps_pot(rab,self.params["rac"]-rab)
        vOut += 2*self.params["kc"]*(rab-(self.params["rac"]/2-x/self.params["c_ho"]))**2
        
        return vOut
    
class SylvesterPot:
    def __init__(self,initialGrid=np.meshgrid(np.arange(-2,2,0.05),np.arange(-3,3,0.05)),\
                 eGS=None):
        self.initialGrid = initialGrid
        self.zz = self.sylvester_pes(self.initialGrid)
        minInds = Utilities.find_local_minimum(self.zz)
        self.minInds = minInds
        
        #Probably useful in general to select the max, so that there's a contour
        #of eGS around the global minimum
        if eGS is None:
            eGS = np.max(self.zz[minInds])
        
        self.potential = Utilities.aux_pot(self.sylvester_pes,eGS)
        self.clipRange = (-1,10)
        
    def sylvester_pes(self,x,y):
        A = -0.8447
        B = -0.2236
        C = 0.1247
        D = -4.468
        E = 0.02194
        F = 0.3041
        G = 0.1687
        H = 0.4388
        I = -4.713 * 10**(-7)
        J = -1.148 * 10**(-5)
        K = 1.687
        L = -3.062 * 10**(-18)
        M = -9.426 * 10**(-6)
        N = -2.851 * 10**(-16)
        O = 2.313 * 10**(-5)
        
        vOut = A + B*x + C*y + D*x**2 + E*x*y + F*y**2 + G*x**3 + H*x**2*y
        vOut += I*x*y**2 + J*y**3 + K*x**4 + L*x**3*y + M*x**2*y**2 + N*x*y**3 + O*y**4
        
        return vOut
    
class FileIO(): #Tried using inheritance here, but I think the static methods
                #messed things up for some reason.
    
    @staticmethod
    def read_from_h5(fName,fDir,baseDir=None):
        if baseDir is None:
            baseDir = Utilities.find_base_dir()
        
        datDictOut = {}
        attrDictOut = {}
        
        h5File = h5py.File(baseDir+fDir+fName,"r")
        allDataSets = [key for key in Utilities.h5_get_keys(h5File) if isinstance(h5File[key],h5py.Dataset)]
        for key in allDataSets:
            datDictOut[key.lstrip("/")] = np.array(h5File[key])
            
        #Does NOT pull out sub-attributes
        for attr in h5File.attrs:
            attrIn = np.array(h5File.attrs[attr])
            #Some data is a scalar, and so would otherwise be stored as a zero-dimensional
            #numpy array. That's just confusing.
            if attrIn.shape == ():
                attrIn = attrIn.reshape((1,))
            attrDictOut[attr.lstrip("/")] = attrIn
        
        h5File.close()
            
        return datDictOut, attrDictOut
    
    @staticmethod
    def new_read_from_h5(fName):
        datDictOut = {}
        attrDictOut = {}
        
        h5File = h5py.File(fName,"r")
        allDataSets = [key for key in Utilities.h5_get_keys(h5File) if isinstance(h5File[key],h5py.Dataset)]
        for key in allDataSets:
            datDictOut[key.lstrip("/")] = np.array(h5File[key])
            
        #Does NOT pull out sub-attributes
        for attr in h5File.attrs:
            attrIn = np.array(h5File.attrs[attr])
            #Some data is a scalar, and so would otherwise be stored as a zero-dimensional
            #numpy array. That's just confusing.
            if attrIn.shape == ():
                attrIn = attrIn.reshape((1,))
            attrDictOut[attr.lstrip("/")] = attrIn
        
        h5File.close()
            
        return datDictOut, attrDictOut
    
    @staticmethod
    def dump_to_hdf5(fname,otpOutputsDict,paramsDictOfDicts):
        h5File = h5py.File(fname+".h5","w")
        for key in otpOutputsDict.keys():
            h5File.create_dataset(key,data=otpOutputsDict[key])
        
        for outerKey in paramsDictOfDicts:
            h5File.create_group(outerKey)
            for key in paramsDictOfDicts[outerKey].keys():
                #Don't know if individual strings are converted automatically
                if isinstance(paramsDictOfDicts[outerKey][key],str):
                    custType = h5py.string_dtype("utf-8")
                else:
                    custType = type(paramsDictOfDicts[outerKey][key])
                h5File[outerKey].attrs.create(key,paramsDictOfDicts[outerKey][key],\
                                                   dtype=custType)
        
        h5File.close()
        
        return None
    
    @staticmethod
    def read_path(fname,returnHeads=False):
        df = pd.read_csv(fname,sep=",",index_col=None,header=None)
        
        firstRow = np.array(df.loc[0])
        try:
            firstRow = firstRow.astype(float)
            ret = np.array(df)
            heads = None
        except ValueError:
            ret = np.array(df.loc[1:]).astype(float)
            heads = df.loc[0]
        
        if returnHeads:
            ret = (ret,heads)
        
        return ret
    
    @staticmethod
    def write_path(fname,path,ndims=2,columnHeads=None):
        if columnHeads is None:
            print("Warning: columnHeads should probably not be None")
            
        #TODO: error handling when path exists
        if ndims not in path.shape:
            sys.exit("Err: path shape is "+str(path.shape)+"; expected one dimension to be "+str(ndims))
            
        if path.shape[0] == ndims:
            path = path.T
            
        df = pd.DataFrame(data=path,columns=columnHeads)
        if columnHeads is None:
            includeHeader = False
        else:
            includeHeader = True
        df.to_csv(fname,sep=",",header=includeHeader,index=False)
        
        return None
    
class CustomLogging():
    """
    A note on parallelization:
        -If one runs multiple instances of the NEB solver concurrently, one
            should output all instances into separate folders *outside* of
            this program (e.g. in the submit script)
        -If one uses a parallelized energy evaluation (e.g. a DFT solver),
            one should be careful when updating the log. Probably, we can
            assume that parallel energy evaluations will have their own
            parallelized logging, and we can simply update the log here once
            we gather all of the energy outputs, without parallelizing anything
    """
    def __init__(self,loggingLevel):
        allowedLogLevels = [None,"output"]#"output" doesn't keep track of intermediate steps
        assert loggingLevel in allowedLogLevels
        self.loggingLevel = loggingLevel
        if self.loggingLevel == "output":
            self.logDict = {}
            self.outputNms = {}
            
            self.stringRep = self.__str__()+"_"+datetime.datetime.now().isoformat()
    
    def update_log(self,strRep,outputTuple,outputNmsTuple,isTuple=True):
        #If returning a single value, set isTuple -> False
        if self.loggingLevel is None:
            return None
        
        if self.loggingLevel == "output":
            if not isTuple:
                outputTuple = (outputTuple,)
                outputNmsTuple = (outputNmsTuple,)
            
            gpName = self.stringRep+"/"+strRep
            if gpName not in self.logDict:
                self.logDict[gpName] = []
                for t in outputTuple:
                    if isinstance(t,np.ndarray):
                        self.logDict[gpName].append(np.expand_dims(t,axis=0))
                    else:
                        self.logDict[gpName].append([t])
                self.outputNms[gpName] = outputNmsTuple
            else:
                assert len(outputTuple) == len(self.logDict[gpName])
                for (tIter,t) in enumerate(outputTuple):
                    if isinstance(t,np.ndarray):
                        self.logDict[gpName][tIter] = \
                            np.concatenate((self.logDict[gpName][tIter],np.expand_dims(t,axis=0)))
                    else:
                        self.logDict[gpName][tIter].append(t)
                        
        return None
    
    def write_log(self,fName,overwrite=False):
        #WARNING: probably doesn't handle anything that isn't a numpy array, although
            #that's almost all that I intend to log at the moment
        #WARNING: does not handle multiple of the same class instance
        if not hasattr(self,"logDict"):
            return None
        
        if not fName.startswith("Logs/"):
            fName = "Logs/"+fName
        os.makedirs("Logs",exist_ok=True)
        
        if (overwrite) and (os.path.isfile(fName)):
            os.remove(fName)
        
        h5File = h5py.File(fName,"a")
        for key in self.logDict.keys():
            splitKey = key.split("/")
            for (sIter,s) in enumerate(splitKey):
                subGp = "/".join(splitKey[:sIter+1])
                if not subGp in h5File:
                    h5File.create_group(subGp)
            for (oIter,outputNm) in enumerate(self.outputNms[key]):
                h5File[key].create_dataset(outputNm,data=self.logDict[key][oIter])
        
        h5File.close()
        
        return None
    
class LineIntegralNeb(CustomLogging):
    def __init__(self,potential,mass,initialPoints,k,kappa,constraintEneg,\
                 targetFunc="action",endSprForce=False,loggingLevel=None,\
                 toCount=["potential","mass"]):
        super().__init__(loggingLevel)#Believe it or not - still unclear what this does...
        
        targetFuncDict = {"action":self._construct_standard_action}
        if targetFunc not in targetFuncDict.keys():
            sys.exit("Err: requested target function "+str(targetFunc)+\
                     "; allowed functions are "+str(targetFuncDict.keys()))
        self.counterFuncDict = {"potential":potential,"mass":mass}
        
        self.counterDict = {}
        #I guess technically allows for counting other things, like self._compute_tangents
        for c in toCount:
            if c not in self.counterFuncDict:
                sys.exit("Err: requested counting of "+c)
            self.counterDict[c] = 0
            setattr(self,c,self.count_evals_wrapper(c))
        
        self.k = k
        self.kappa = kappa
        self.constraintEneg = constraintEneg
        
        self.nCoords, self.nPts = initialPoints.shape
        print(self.nCoords)
        
        #Only set the potential and mass by name
        for f in ["potential","mass"]:
            if not hasattr(self,f):
                setattr(self,f,self.counterFuncDict[f])
        
        self.initialPoints = initialPoints
        self.endSprForce = endSprForce
        
        #Is an approximation of the integral
        self.target_func = targetFuncDict[targetFunc]()
        
    def __str__(self):
        #Kind of useful b/c it exists regardless of the state of self.__init__().
        #For instance, I would otherwise have to define this before calling
        #super().__init__(), or else it doesn't exist
        return "LineIntegralNeb"
    
    def count_evals_wrapper(self,counterString):
        def func_out(*args,**kwargs):
            func = self.counterFuncDict[counterString]
            self.counterDict[counterString] += 1
            return func(*args,**kwargs)
        return func_out
    
    def _construct_standard_action(self):
        def standard_action(path,enegs=None,masses=None):
            #In case I feed in a tuple, as seems likely given how I handle passing
            #coordinates around
            if not isinstance(path,np.ndarray):
                path = np.array(path)
                
            if enegs is None:
                enegs = self.potential(path)
            assert enegs.shape[0] == path.shape[1]
            
            #Will add error checking here later - still not general with a nonconstant mass
            if masses is None:
                masses = self.mass(path)
                
            if masses.shape != (self.nPts, self.nCoords, self.nCoords):
                sys.exit("Err: mass.shape = "+str(masses.shape)+\
                         ", required shape = "+str((self.nPts, self.nCoords, self.nCoords)))
            
            retVal = 0
            for ptIter in np.arange(1,self.nPts):
                coordDiff = path[:,ptIter] - path[:,ptIter-1]
                dist = np.dot(coordDiff,np.dot(masses[ptIter],coordDiff))
                retVal += np.sqrt(2*enegs[ptIter]*dist)
            # print(enegs)
            if np.count_nonzero(enegs<0) > 0:
                print("path:")
                print(path)
                sys.exit("...")
                
            if np.count_nonzero(np.isinf(enegs)):
                print("path:")
                print(path)
                sys.exit("...")
                
            return retVal, enegs, masses
        return standard_action
    
    def _compute_tangents(self,currentPts,energies):
        strRep = "_compute_tangents"
        
        tangents = np.zeros((self.nCoords,self.nPts))
        for ptIter in range(1,self.nPts-1): #Range selected to exclude endpoints
            tp = currentPts[:,ptIter+1] - currentPts[:,ptIter]
            tm = currentPts[:,ptIter] - currentPts[:,ptIter-1]
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
            # elif energies[ptIter+1] <= energies[ptIter-1]:
            else:
                tangents[:,ptIter] = tp*dVMin + tm*dVMax
                
            #Normalizing vectors
            tangents[:,ptIter] = tangents[:,ptIter]/np.sqrt(np.dot(tangents[:,ptIter],tangents[:,ptIter]))
        
        ret = tangents
        outputNms = "tangents"
        self.update_log(strRep,ret,outputNms,isTuple=False)
        
        return ret
    
    def _negative_gradient(self,points,enegsOnPath,massesOnPath):
        strRep = "_negative_gradient"
        
        eps = 10**(-8)
        
        trueForce = np.zeros(points.shape)
        negIntegGrad = np.zeros(points.shape)
        
        actionOnPath, _, _ = self.target_func(points,enegs=enegsOnPath,masses=massesOnPath)
        
        for ptIter in range(self.nPts):
            for coordIter in range(self.nCoords):
                steps = points.copy()
                steps[coordIter,ptIter] += eps
                
                enegsAtStep = enegsOnPath.copy()
                enegsAtStep[ptIter] = self.potential(steps[:,ptIter])
                
                massesAtStep = massesOnPath.copy()
                massesAtStep[ptIter] = self.mass(steps[:,ptIter])
                
                actionAtStep, _, _ = self.target_func(steps,enegs=enegsAtStep,\
                                                      masses=massesAtStep)
                trueForce[coordIter,ptIter] = (enegsAtStep[ptIter] - enegsOnPath[ptIter])/eps
                negIntegGrad[coordIter,ptIter] = (actionAtStep - actionOnPath)/eps
        
        #Want the negative gradient of both terms
        trueForce = -trueForce
        negIntegGrad = -negIntegGrad
        
        ret = (trueForce, negIntegGrad)
        outputNms = ("trueForce", "negIntegGrad")
        self.update_log(strRep,ret,outputNms)
        
        return ret
    
    def _spring_force(self,points,tangents):
        strRep = "_spring_force"
        
        diffArr = np.array([points[:,i+1] - points[:,i] for i in range(points.shape[1]-1)]).T
        diffScal = np.array([np.linalg.norm(diffArr[:,i]) for i in range(diffArr.shape[1])])
        
        springForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            springForce[:,i] = self.k*(diffScal[i] - diffScal[i-1])*tangents[:,i]
            
        if not self.endSprForce:
            springForce[:,0] = np.zeros(self.nCoords)
            springForce[:,-1] = np.zeros(self.nCoords)
        else:
            springForce[:,0] = self.k*diffArr[:,0]
            springForce[:,-1] = -self.k*diffArr[:,-1] #Added minus sign here to fix endpoint behavior
        
        ret = springForce
        outputNms = "springForce"
        self.update_log(strRep,ret,outputNms,isTuple=False)
        
        return ret
    
    def compute_force(self,points):
        strRep = "compute_force"
        
        integVal, energies, masses = self.target_func(points)
        
        tangents = self._compute_tangents(points,energies)
        trueForce, negIntegGrad = self._negative_gradient(points,energies,masses)
        
        #Note: don't care about the tangents on the endpoints; they don't show up
        #in the net force
        perpForce = negIntegGrad - tangents*(np.array([np.dot(negIntegGrad[:,i],tangents[:,i]) \
                                                       for i in range(points.shape[1])]))
        springForce = self._spring_force(points,tangents)
        
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            netForce[:,i] = perpForce[:,i] + springForce[:,i]
            
        if np.array_equal(trueForce[:,0],np.zeros(self.nCoords)):
            normForce = np.zeros(self.nCoords)
        else:
            normForce = trueForce[:,0]/np.linalg.norm(trueForce[:,0])
        netForce[:,0] = springForce[:,0] - \
            (np.dot(springForce[:,0],normForce)-\
              self.kappa*(energies[0]-self.constraintEneg))*normForce
            
        if np.array_equal(trueForce[:,-1],np.zeros(self.nCoords)):
            normForce = np.zeros(self.nCoords)
        else:
            normForce = trueForce[:,-1]/np.linalg.norm(trueForce[:,-1])
        netForce[:,-1] = springForce[:,-1] - \
            (np.dot(springForce[:,-1],normForce)-\
              self.kappa*(energies[-1]-self.constraintEneg))*normForce
        
        ret = netForce
        outputNms = "netForce"
        self.update_log(strRep,ret,outputNms,isTuple=False)
        
        #Testing for now
        netForce[:,0] = np.zeros(self.nCoords)
        
        return ret
    
class MinimizationAlgorithms(CustomLogging):
    def __init__(self,lnebObj,initialPoints=None,loggingLevel=None,fig=None,ax=None):
        super().__init__(loggingLevel)
        
        self.lneb = lnebObj
        self.action_func = self.lneb.target_func #Idk why I did this
        self.fig = fig
        self.ax = ax
        
        if hasattr(self.lneb,"nCoords"):
            self.nCoords = self.lneb.nCoords
        elif hasattr(self.lneb,"nDims"):
            self.nCoords = self.lneb.nDims
        else:
            raise AttributeError("self.lneb has no accepted attribute nCoords")
            
        if hasattr(self.lneb,"initialPoints"):
            self.initialPoints = self.lneb.initialPoints
        else:
            self.initialPoints = initialPoints
        
        if self.initialPoints is None:
            raise AttributeError("initialPoints not supplied")
    
    def verlet_minimization(self,maxIters=1000,tStep=0.05):
        strRep = "verlet_minimization"
        
        allPts = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        
        allPts[0,:,:] = self.initialPoints
        
        for step in range(0,maxIters):
            force = self.lneb.compute_force(allPts[step,:,:])
            allForces[step,:,:] = force
            allPts[step+1,:,:] = allPts[step,:,:] + 1/2*force*tStep**2
            
        actions = np.array([self.action_func(pts)[0] for pts in allPts[:]])
        
        ret = (allPts, allForces, actions)
        outputNms = ("allPts","allForces","actions")
        self.update_log(strRep,ret,outputNms)
        
        return ret
    
    def verlet_minimization_v2(self,maxIters=1000,tStep=0.05):
        strRep = "verlet_minimization_v2"
        
        allPts = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        allVelocities = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        
        allPts[0,:,:] = self.initialPoints
        f = self.lneb.compute_force(allPts[0,:,:])
        if f.shape != (self.nCoords,self.lneb.nPts):
            f = f.T
        allForces[0,:,:] = f
        
        for step in range(0,maxIters):
            #Velocity update taken from "Classical and Quantum Dynamics in Condensed Phase Simulations",
            #page 397
            for ptIter in range(self.lneb.nPts):
                product = np.dot(allVelocities[step,:,ptIter],allForces[step,:,ptIter])
                if product > 0:
                    vProj = \
                        product*allForces[step,:,ptIter]/np.dot(allForces[step,:,ptIter],allForces[step,:,ptIter])
                else:
                    vProj = np.zeros(self.nCoords)
                allVelocities[step+1,:,ptIter] = vProj + allForces[step,:,ptIter]*tStep
            allPts[step+1] = allPts[step] + allVelocities[step+1]*tStep+1/2*allForces[step]*tStep**2
            f = self.lneb.compute_force(allPts[step+1])
            if f.shape != (self.nCoords,self.lneb.nPts):
                f = f.T
            allForces[step+1] = f
            
            # if self.ax is not None:
            #     self.ax.plot(allPts[step,0],allPts[step,1])
            #     self.fig.show()
                
        try:
            actions = np.array([self.action_func(pts)[0] for pts in allPts[:]])
        except TypeError: #For when I'm transitioning to the py_neb code
            actions = np.array([self.action_func(pts.T,self.lneb.potential)[0] for pts in allPts[:]])
        
        ret = (allPts, allVelocities, allForces, actions)
        outputNms = ("allPts","allVelocities","allForces","actions")
        self.update_log(strRep,ret,outputNms)
        
        return ret
    
    def verlet_minimization_with_fire(self,maxIters=1000,dtMin=0.1,minFire=10,dtMax=10,\
                                      finc=1.1,fdec=0.5,fadec=0.99,alphaInit=0.1,\
                                      trimNans=True):
        strRep = "verlet_minimization_with_fire"
        
        fireSteps = 0
        tStep = dtMin
        alpha = alphaInit
        
        allPts = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        allVelocities = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.nCoords,self.lneb.nPts))
        
        allPts[0,:,:] = self.initialPoints
        allForces[0,:,:] = self.lneb.compute_force(allPts[0,:,:])
        
        for step in range(0,maxIters):
            #Velocity update taken from FIRE algorithm (see e.g. arXiv:1908.02038v4)
            for ptIter in range(self.lneb.nPts):
                product = np.dot(allVelocities[step,:,ptIter],allForces[step,:,ptIter])
                if product > 0:
                    vProj = (1. - alpha)*allVelocities[step,:,ptIter]
                    vProj += alpha*allForces[step,:,ptIter] * np.linalg.norm(allVelocities[step,:,ptIter])\
                        /np.linalg.norm(allForces[step,:,ptIter])
                    if (fireSteps > minFire):
                        tStep = min(tStep*finc,dtMax)
                        alpha = alpha * fadec
                    fireSteps += 1
                else:
                    vProj = np.zeros(self.nCoords)
                    alpha = alphaInit
                    tStep = max(tStep*fdec,dtMin)
                    fireSteps = 0
                allVelocities[step+1,:,ptIter] = vProj + allForces[step,:,ptIter]*tStep
            allPts[step+1] = allPts[step] + allVelocities[step+1]*tStep+1/2*allForces[step]*tStep**2
            allForces[step+1] = self.lneb.compute_force(allPts[step+1])
                
        actions = np.array([self.action_func(pts)[0] for pts in allPts[:]])
        
        # numberOfNans = np.count_nonzero(np.isnan(allPts))
        # if numberOfNans > 0:
        #     print("Found "+str(numberOfNans)+" NaNs in allPts")
        
        #I want to see where the path ends up going off
        if trimNans:
            trimIter = allPts.shape[0]
            for i in range(allPts.shape[0]):
                if np.count_nonzero(np.isnan(allPts[i])) > 0:
                    trimIter = i
                    break
            allPts = allPts[:trimIter]
            allVelocities = allVelocities[:trimIter]
            allForces = allForces[:trimIter]
            actions = actions[:trimIter]
        
        ret = (allPts, allVelocities, allForces, actions)
        outputNms = ("allPts","allVelocities","allForces","actions")
        self.update_log(strRep,ret,outputNms)
        
        return allPts, allVelocities, allForces, actions

def uranium_test():
    startTime = time.time()
    
    fDir = "Daniels_Code/"
    fName = "252U_PES.h5"
    dsets, attrs = FileIO.read_from_h5(fName,fDir)
    
    #Only getting the unique values
    q20Vals = dsets["Q20"][:,0]
    q30Vals = dsets["Q30"][0]
    
    zz = dsets["PES"] - np.min(dsets["PES"])
    
    interpPot = interpnd_wrapper((q20Vals,q30Vals),zz)
    potential = Utilities.aux_pot(interpPot,0)
    
    # otherPot = CustomInterp2d(q20Vals,q30Vals,zz.T,kind="linear")
    # potential2 = Utilities.aux_pot(otherPot.potential,0)
    
    # thirdPot = CustomInterp2d(q20Vals,q30Vals,zz.T,kind="cubic")
    # potential3 = Utilities.aux_pot(thirdPot.potential,0)
    
    # fourthPot = CustomInterp2d(q20Vals,q30Vals,zz.T,kind="quintic")
    # potential4 = Utilities.aux_pot(fourthPot.potential,0)
    
    nPts = 22
    k = 10
    kappa = 20
    
    initialPoints = np.array((np.linspace(26,190,nPts),np.linspace(1,16,nPts)))
    
    initialEnegs = potential(initialPoints)
    print(initialEnegs)
    constraintEneg = initialEnegs[0]
    
    print("Constraining to energy %.3f" % constraintEneg)
    
    maxIters = 250
    
    lneb = LineIntegralNeb(potential,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    minObj = MinimizationAlgorithms(lneb)
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters)
    
    # lneb2 = LineIntegralNeb(potential2,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    # minObj2 = MinimizationAlgorithms(lneb2)
    # allPts2, allVelocities2, allForces2, actions2 = \
    #     minObj2.verlet_minimization_v2(maxIters=maxIters)
        
    # lneb3 = LineIntegralNeb(potential3,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    # minObj3 = MinimizationAlgorithms(lneb3)
    # allPts3, allVelocities3, allForces3, actions3 = \
    #     minObj3.verlet_minimization_v2(maxIters=maxIters)
        
    # lneb4 = LineIntegralNeb(potential4,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    # minObj4 = MinimizationAlgorithms(lneb4)
    # allPts4, allVelocities4, allForces4, actions4 = \
    #     minObj4.verlet_minimization_v2(maxIters=maxIters)
        
    """   Plotting   """
    fig, ax = plt.subplots()
    ax.plot(actions,label="ND Linear",color="blue")
    # ax.plot(actions2,label="2D Linear",color="green")
    # ax.plot(actions3,label="2D Cubic",color="red")
    # ax.plot(actions4,label="2D Quintic",color="orange")
    
    ax.set(xlabel="Iteration",ylabel="Action")
    ax.legend()
    # fig.savefig("Runs/"+str(int(startTime))+"_Action.pdf")
    
    cbarRange = (-5,30)
    fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],zz)
    ax.contour(dsets["Q20"],dsets["Q30"],zz,levels=[constraintEneg],\
                colors=["black"])
    
    ax.plot(allPts[-1,0],allPts[-1,1],marker=".",label="ND Linear",color="blue")
    # ax.plot(allPts2[-1,0],allPts2[-1,1],marker=".",label="2D Linear",color="green")
    # ax.plot(allPts3[-1,0],allPts3[-1,1],marker=".",label="2D Cubic",color="red")
    # ax.plot(allPts4[-1,0],allPts4[-1,1],marker=".",label="2D Quintic",color="orange")
    
    ax.legend(loc="upper left")
    ax.set_xlim(min(q20Vals),max(q20Vals))
    ax.set_ylim(min(q30Vals),max(q30Vals))
    ax.set(title=r"${}^{252}$U PES")
        
    # fig.savefig("Runs/"+str(int(startTime))+".pdf")
    
    # FileIO.write_path("ND_Linear.txt",allPts[-1])
    # FileIO.write_path("2D_Linear.txt",allPts2[-1])
    # FileIO.write_path("2D_Cubic.txt",allPts3[-1])
    # FileIO.write_path("2D_Quintic.txt",allPts4[-1])
    
    return None

def uranium_pyneb_test():
    fDir = "Daniels_Code/"
    fName = "252U_PES.h5"
    dsets, attrs = FileIO.read_from_h5(fName,fDir)
    
    #Only getting the unique values
    q20Vals = dsets["Q20"][:,0]
    q30Vals = dsets["Q30"][0]
    
    zz = dsets["PES"]# - np.min(dsets["PES"])
    additionalShift = 0#-1.
    
    interpPot = interpnd_wrapper((q20Vals,q30Vals),zz)
    potential = Utilities.aux_pot(interpPot,additionalShift)
    
    # flattenedPts = np.array([dsets["Q20"],dsets["Q30"]]).reshape((2,-1))
    # enegs = potential(flattenedPts)
    
    # print("Validating interpolator. Mean difference: "+str(np.mean(enegs.reshape(zz.shape) - zz)))
    
    # otherPot = CustomInterp2d(q20Vals,q30Vals,zz.T,kind="linear")
    # potential2 = Utilities.aux_pot(otherPot.potential,0)
    
    # thirdPot = CustomInterp2d(q20Vals,q30Vals,zz.T,kind="cubic")
    # potential3 = Utilities.aux_pot(thirdPot.potential,0)
    
    # fourthPot = CustomInterp2d(q20Vals,q30Vals,zz.T,kind="quintic")
    # potential4 = Utilities.aux_pot(fourthPot.potential,0)
    
    nPts = 22
    k = 10
    kappa = 20
    
    initialPoints = np.array((np.linspace(26,190,nPts),np.linspace(1,16,nPts)))
    
    initialEnegs = potential(initialPoints)
    print(initialEnegs)
    constraintEneg = initialEnegs[0]
    
    print("Constraining to energy %.3f" % constraintEneg)
    
    maxIters = 750
    
    lap = py_neb.LeastActionPath(potential,nPts,2,nebParams={"k":k,"kappa":kappa,"constraintEneg":constraintEneg})
    # lneb = LineIntegralNeb(potential,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    minObj = MinimizationAlgorithms(lap,initialPoints=initialPoints)
    
    startTime = time.time()
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters)
    endTime = time.time()
    print("Run time: "+str(endTime-startTime))
    # print(actions)
    # lneb2 = LineIntegralNeb(potential2,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    # minObj2 = MinimizationAlgorithms(lneb2)
    # allPts2, allVelocities2, allForces2, actions2 = \
    #     minObj2.verlet_minimization_v2(maxIters=maxIters)
        
    # lneb3 = LineIntegralNeb(potential3,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    # minObj3 = MinimizationAlgorithms(lneb3)
    # allPts3, allVelocities3, allForces3, actions3 = \
    #     minObj3.verlet_minimization_v2(maxIters=maxIters)
        
    # lneb4 = LineIntegralNeb(potential4,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    # minObj4 = MinimizationAlgorithms(lneb4)
    # allPts4, allVelocities4, allForces4, actions4 = \
    #     minObj4.verlet_minimization_v2(maxIters=maxIters)
        
    """   Plotting   """
    fig, ax = plt.subplots()
    ax.plot(actions,label="ND Linear",color="blue")
    # ax.plot(actions2,label="2D Linear",color="green")
    # ax.plot(actions3,label="2D Cubic",color="red")
    # ax.plot(actions4,label="2D Quintic",color="orange")
    
    ax.set(xlabel="Iteration",ylabel="Action")
    ax.legend()
    # fig.savefig("Runs/"+str(int(startTime))+"_Action.pdf")
    
    cbarRange = (-5,30)
    # fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],zz)
    # ax.contour(dsets["Q20"],dsets["Q30"],zz,levels=[constraintEneg],\
    #            colors=["black"])
    
    fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],zz)
    ax.contour(dsets["Q20"],dsets["Q30"],zz,levels=[constraintEneg+additionalShift],\
               colors=["black"])
    
    ax.plot(allPts[-1,0],allPts[-1,1],marker=".",label="252U_ND_Linear",color="blue")
    # ax.plot(allPts2[-1,0],allPts2[-1,1],marker=".",label="2D Linear",color="green")
    # ax.plot(allPts3[-1,0],allPts3[-1,1],marker=".",label="2D Cubic",color="red")
    # ax.plot(allPts4[-1,0],allPts4[-1,1],marker=".",label="2D Quintic",color="orange")
    
    ax.legend(loc="upper left")
    ax.set_xlim(min(q20Vals),max(q20Vals))
    ax.set_ylim(min(q30Vals),max(q30Vals))
    ax.set(title=r"${}^{252}$U PES")
        
    fig.savefig("252U.pdf")
    
    FileIO.write_path("252U_ND_Linear.txt",allPts[-1],columnHeads=["Q20","Q30"])
    # FileIO.write_path("2D_Linear.txt",allPts2[-1])
    # FileIO.write_path("2D_Cubic.txt",allPts3[-1])
    # FileIO.write_path("2D_Quintic.txt",allPts4[-1])
    
    # print(allPts[-1])
    eGS = potential(allPts[-1])[0]
    print(eGS)
    potential_with_zero_egs = Utilities.aux_pot(potential,eGS,tol=10**(-4))
    # print(potential_with_zero_egs(allPts[-1]))
    
    # finalAction, _, _ = py_neb.action(allPts[-1].T,potential_with_zero_egs)
    print("Action shifted up: "+str(actions[-1]))
    # print("Action unshifted: "+str(finalAction))
    
    return None

def u232_test():
    fIn = "..//PES/232U.h5"
    dsets, attrs = FileIO.new_read_from_h5(fIn)
    
    coordStrs = ["Q20","Q30"]
    
    uniqueCoords = [np.unique(dsets[key]) for key in coordStrs]
    
    gridShape = [len(np.unique(dsets[key])) for key in coordStrs]
    
    coordMesh = tuple([dsets[key].reshape(gridShape) for key in coordStrs])
    zz = dsets["PES"].reshape(gridShape)
    
    potential = interpnd_wrapper(uniqueCoords,zz)
    
    #Finding initial path
    gsLoc = np.array([attrs["Ground_State"][key] for key in coordStrs]).flatten()
    eGS = potential(gsLoc)
    
    allConts = Utilities.find_approximate_contours(coordMesh,zz)
    outerContour = allConts[0][1]
    
    nPts = 22
    initPath = np.array((np.linspace(gsLoc[0],300,nPts),np.linspace(gsLoc[1],32,nPts)))
    
    f, a = Utilities.standard_pes(*coordMesh,zz)
    a.contour(*coordMesh,zz,levels=[eGS],colors=["black"])
    
    lap = py_neb.LeastActionPath(potential,22,2,\
                                 nebParams={"k":20,"kappa":10},\
                                 endpointSpringForce=(False,True),\
                                 endpointHarmonicForce=(False,True))
            
    maxIters = 750
    tStep = 0.05
    
    minObj = MinimizationAlgorithms(lap,initialPoints=initPath)
    t0 = time.time()
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters,tStep=tStep)
    t1 = time.time()
    a.plot(allPts[-1,0],allPts[-1,1],marker=".",label="Slow Interpolator")
    
    print("Fast Interpolator time: "+str(t1 - t0))
    print("Fast Interpolator action: "+str(actions[-1]))
    
    actionFig, actionAx = plt.subplots()
    actionAx.plot(actions,label="Slow Interpolator")
    
    potential = GridInterpWithBoundary(uniqueCoords,zz)
    potential = Utilities.aux_pot(potential,0,tol=1)
    
    
    lap = py_neb.LeastActionPath(potential,22,2,\
                                 nebParams={"k":20,"kappa":10},\
                                 endpointSpringForce=(False,True),\
                                 endpointHarmonicForce=(False,True))
        
    minObj = MinimizationAlgorithms(lap,initialPoints=initPath)
    t0 = time.time()
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters,tStep=tStep)
    t1 = time.time()
    
    a.plot(allPts[-1,0],allPts[-1,1],marker=".",label="Fast Interpolator")
    a.legend(loc="upper left")
    
    actionAx.plot(actions,label="Fast Interpolator")
    actionAx.legend()
    
    print("Fast Interpolator time: "+str(t1 - t0))
    print("Fast Interpolator action: "+str(actions[-1]))
    
    return None

def grid_interp_test():
    fIn = "..//PES/232U.h5"
    dsets, attrs = FileIO.new_read_from_h5(fIn)
    
    coordStrs = ["Q20","Q30"]
    
    uniqueCoords = [np.unique(dsets[key]) for key in coordStrs]
    gridShape = [len(np.unique(dsets[key])) for key in coordStrs]
    
    coordMesh = tuple([dsets[key].reshape(gridShape) for key in coordStrs])
    zz = dsets["PES"].reshape(gridShape)
    
    g = py_neb.GridInterpWithBoundary(uniqueCoords,zz)
    points = np.array([[0,0],[0.7,0.364],[-1,0]])
    # print(dsets["Q20"])
    # print(dsets["Q30"])
    print(g(points))
    
    return None


def plutonium_pes_slices():
    dsets, _ = FileIO.read_from_h5("240Pu.h5","/",baseDir=os.getcwd())
    coords = ["Q20","Q30","pairing"]
    uniqueCoords = [np.unique(dsets[coord]) for coord in coords]
    desiredShape = np.array([len(c) for c in uniqueCoords],dtype=int)
    
    reshapedData = {key:dsets[key].reshape(desiredShape) for key in dsets.keys()}
    cmesh = tuple([reshapedData[coord] for coord in coords])
    zz = reshapedData["E_HFB"]
    minInds = Utilities.find_local_minimum(zz)
    allowedInds = tuple(np.array([inds[zz[minInds]!=-1760] for inds in minInds]))
    
    spac = desiredShape//5 #Integer division
    # print([m[allowedInds] for m in cmesh])
    gsInds = Utilities.extract_gs_inds(allowedInds,cmesh,zz)
    eGS = zz[gsInds]
    print([c[gsInds] for c in cmesh])
    print(eGS)
    
    fig, ax = plt.subplots(nrows=3,ncols=5,figsize=(20,12))
    for i in range(5):
        for c in range(3):
            xInd = (c+1) % 3
            yInd = (c+2) % 3
            inds = tuple(c*[slice(None)] + [spac[c]*i] + (2-c)*[slice(None)])
            ax[c,i].contourf(cmesh[xInd][inds],cmesh[yInd][inds],reshapedData["E_HFB"][inds])
            ax[c,i].scatter(cmesh[xInd][allowedInds],cmesh[yInd][allowedInds],color="red",marker="x")
            ax[c,i].scatter(cmesh[xInd][gsInds],cmesh[yInd][gsInds],s=200,color="green",marker="*")
            
            xPos = 0.25*(np.max(dsets[coords[xInd]])-np.min(dsets[coords[xInd]]))+np.min(dsets[coords[xInd]])
            yPos = 0.25*(np.max(dsets[coords[yInd]])-np.min(dsets[coords[yInd]]))+np.min(dsets[coords[yInd]])
            ax[c,i].axhline(yPos,color="black")
            ax[c,i].axvline(xPos,color="black")
            
            ax[c,i].contour(cmesh[xInd][inds],cmesh[yInd][inds],reshapedData["E_HFB"][inds],\
                            levels=[eGS],colors=["white"])
            # print(cmesh[c][inds])
            # ax[c,i].set(xlim=(np.min(dsets[coords[xInd]]),np.max(dsets[coords[xInd]])),\
            #             ylim=(np.min(dsets[coords[yInd]]),np.max(dsets[coords[yInd]])),\
            #             xlabel=coords[xInd],ylabel=coords[yInd],\
            #             title=coords[c]+" = "+str(cmesh[c][inds][0,0]))
            ax[c,i].set(xlabel=coords[xInd],ylabel=coords[yInd],\
                        title=coords[c]+" = "+str(cmesh[c][inds][0,0]))
            
    fig.tight_layout()
    fig.savefig("240Pu_PES_Slices.pdf")
    
    return None

def plutonium_test():
    sylvesterPath = FileIO.read_path("240Pu_Sylvester_Path.txt")
    newPath = FileIO.read_path("3DPUMAL_NEW.txt")
    
    dsets, _ = FileIO.read_from_h5("240Pu.h5","/",baseDir=os.getcwd())
    coords = ["Q20","Q30","pairing"]
    uniqueCoords = [np.unique(dsets[coord]) for coord in coords]
    desiredShape = np.array([len(c) for c in uniqueCoords],dtype=int)
    
    reshapedData = {key:dsets[key].reshape(desiredShape) for key in dsets.keys()}
    cmesh = tuple([reshapedData[coord] for coord in coords])
    zz = reshapedData["E_HFB"]
    minInds = Utilities.find_local_minimum(zz)
    allowedInds = tuple(np.array([inds[zz[minInds]!=-1760] for inds in minInds]))
    
    gsInds = Utilities.extract_gs_inds(allowedInds,cmesh,zz)
    gsLoc = [c[gsInds] for c in cmesh]
    absMin = zz.min()
    zz = zz - absMin
    eGS = zz[gsInds]
    # zz = zz - eGS
    
    interp_eneg = interpnd_wrapper(uniqueCoords,zz)
    
    inertiaKeysArr = np.array(['B2020', 'B2030', 'B20pair', 'B2030','B3030', \
                               'B30pair', 'B20pair','B30pair','Bpairpair'])
    inertiaFuncsArr = np.array([interpnd_wrapper(uniqueCoords,reshapedData[key])\
                                for key in inertiaKeysArr],dtype=object).reshape((3,3))
    
    inertia_func = mass_array_func(inertiaFuncsArr)
    
    allContours = Utilities.find_approximate_contours(cmesh,zz,eneg=eGS)
    outerContours = []
    outerContoursFullDimension = []
    for (cIter,cont) in enumerate(allContours):
        fillVal = cmesh[2][0,0,cIter]

        lengths = np.array([c.shape[0] for c in cont])
        correctInd = np.argmax(lengths)
        outerContours.append(cont[correctInd])
        
        appArr = np.hstack((cont[correctInd],fillVal*np.ones((lengths[correctInd],1))))
        outerContoursFullDimension.append(appArr)
    
    pathList = [Utilities.new_init_on_contour(cont,gsLoc) for \
                cont in outerContoursFullDimension]
    
    tStep = 0.05
    k = 1
    kappa = 10
    lneb = LineIntegralNeb(interp_eneg,inertia_func,pathList[0],k,kappa,eGS,\
                           endSprForce=True,loggingLevel="output")
    minObj = MinimizationAlgorithms(lneb,loggingLevel="output")
    maxIters = 100
    allPts, allForces, allVelocities, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters,tStep=tStep)
    
    lneb2 = LineIntegralNeb(interp_eneg,inertia_func,pathList[0],k,kappa,eGS,\
                            endSprForce=True,loggingLevel="output")
    
    minObj2 = MinimizationAlgorithms(lneb,loggingLevel="output")
    maxIters = 100
    allPts2, allForces2, actions2 = \
        minObj2.verlet_minimization(maxIters=maxIters,tStep=tStep)
    
    logFile = "Test_Log.h5"
    lneb.write_log(logFile,overwrite=True)
    minObj.write_log(logFile)
    
    lneb2.write_log(logFile)
    minObj2.write_log(logFile)
    
    dummy1 = LineIntegralNeb(interp_eneg,inertia_func,sylvesterPath.T,1,1,0,toCount=[])
    sylvesterAction, _, _ = dummy1.target_func(sylvesterPath.T)
    
    dummy2 = LineIntegralNeb(interp_eneg,inertia_func,newPath.T,1,1,0,toCount=[])
    newAction, _, _ = dummy1.target_func(newPath.T)
    
    fig, ax = plt.subplots()
    ax.plot(actions,color="blue",label="My Action (With Velocity): %.1f" % actions[-1])
    ax.plot(actions2,color="black",label="My Action (Without Velocity): %.1f" % actions2[-1])
    ax.axhline(sylvesterAction,color="red",label="Old Sylvester Action: {:.1}".format(sylvesterAction))
    ax.axhline(newAction,color="green",label="New Sylvester Action: %.1f" % newAction)
    ax.set(xlabel="Iteration",ylabel="Action")
    ax.legend()
    fig.savefig("240Pu_NEB_Action.pdf")
    
    fig, ax = plt.subplots()
    ax.contourf(cmesh[0][:,:,0],cmesh[1][:,:,0],zz[:,:,0])
    ax.plot(allPts[0,0],allPts[0,1],marker=".",color="grey",label="Initial Path")
    ax.plot(allPts[-1,0],allPts[-1,1],marker=".",color="blue",label="Final Path (With Velocity)")
    ax.plot(allPts2[-1,0],allPts2[-1,1],marker=".",color="black",label="Final Path (Without Velocity)")
    ax.plot(sylvesterPath[:,0],sylvesterPath[:,1],marker="x",color="red",label="Old Sylvester Path")
    ax.plot(newPath[:,0],newPath[:,1],marker="x",color="green",label="New Sylvester Path")
    ax.legend(loc="lower right")
    ax.set(xlabel="Q20",ylabel="Q30",title=r"Pairing Slice $\lambda_2=0$")
    fig.savefig("240Pu_NEB.pdf")
    # anotherLneb = LineIntegralNeb(interp_eneg,inertia_func,pathList[1],5,20,0,\
    #                               endSprForce=True,loggingLevel="output")
    # minObj = MinimizationAlgorithms(anotherLneb,loggingLevel="output")
    # maxIters = 10
    # allPts, allForces, actions = \
    #     minObj.verlet_minimization(maxIters=maxIters,tStep=0.01)
    FileIO.write_path("240Pu_Test_Path_Velocity.csv",allPts[-1],ndims=3,\
                      columnHeads=["Q20","Q30","pairing"])
    FileIO.write_path("240Pu_Test_Path_No_Velocity.csv",allPts2[-1],ndims=3,\
                      columnHeads=["Q20","Q30","pairing"])
    
    # anotherLneb.write_log(logFile)
    # minObj.write_log(logFile)
    return None

def plutonium_endpoint_test():
    dsets, _ = FileIO.read_from_h5("240Pu.h5","/",baseDir=os.getcwd())
    coords = ["Q20","Q30","pairing"]
    uniqueCoords = [np.unique(dsets[coord]) for coord in coords]
    desiredShape = np.array([len(c) for c in uniqueCoords],dtype=int)
    
    reshapedData = {key:dsets[key].reshape(desiredShape) for key in dsets.keys()}
    cmesh = tuple([reshapedData[coord] for coord in coords])
    zz = reshapedData["E_HFB"]
    minInds = Utilities.find_local_minimum(zz)
    allowedInds = tuple(np.array([inds[zz[minInds]!=-1760] for inds in minInds]))
    
    gsInds = Utilities.extract_gs_inds(allowedInds,cmesh,zz)
    gsLoc = [c[gsInds] for c in cmesh]
    eGS = zz[gsInds]
    zz = zz - eGS
    
    interp_eneg = interpnd_wrapper(uniqueCoords,zz)
    
    inertiaKeysArr = np.array(['B2020', 'B2030', 'B20pair', 'B2030','B3030', \
                               'B30pair', 'B20pair','B30pair','Bpairpair'])
    inertiaFuncsArr = np.array([interpnd_wrapper(uniqueCoords,reshapedData[key])\
                                for key in inertiaKeysArr],dtype=object).reshape((3,3))
    
    inertia_func = mass_array_func(inertiaFuncsArr)
    
    allContours = Utilities.find_approximate_contours(cmesh,zz)
    outerContours = []
    outerContoursFullDimension = []
    for (cIter,cont) in enumerate(allContours):
        fillVal = cmesh[2][0,0,cIter]

        lengths = np.array([c.shape[0] for c in cont])
        correctInd = np.argmax(lengths)
        outerContours.append(cont[correctInd])
        
        appArr = np.hstack((cont[correctInd],fillVal*np.ones((lengths[correctInd],1))))
        outerContoursFullDimension.append(appArr)
    
    pathList = [Utilities.new_init_on_contour(cont,gsLoc) for \
                cont in outerContoursFullDimension]
    
    logFile = "Endpoint_Testing.h5"
    for (pathIter, path) in enumerate(pathList):
        lneb = LineIntegralNeb(interp_eneg,inertia_func,path,5,20,0,\
                               endSprForce=True,loggingLevel="output")
        
        minObj = MinimizationAlgorithms(lneb,loggingLevel="output")
        maxIters = 1000
        allPts, allForces, actions = \
            minObj.verlet_minimization(maxIters=maxIters,tStep=0.01)
        
        lneb.write_log(logFile)
        minObj.write_log(logFile)
        
        FileIO.write_path("./Paths/Endpoint_Testing/Slice_"+str(pathIter)+"_Start.txt",\
                          allPts[0],ndims=3,columnHeads=["Q20","Q30","pairing"])
        FileIO.write_path("./Paths/Endpoint_Testing/Slice_"+str(pathIter)+"_Endpoint.txt",\
                          allPts[-1],ndims=3,columnHeads=["Q20","Q30","pairing"])
        fig, ax = plt.subplots()
        ax.plot(actions)
        
        ax.set(xlabel="Iteration",ylabel="Action")
        fig.savefig("Endpoint_Testing_Plots/Slice_"+str(pathIter)+".pdf")
        plt.show()
    
    return None

def nd_contour_test():
    dsets, _ = FileIO.read_from_h5("240Pu.h5","/",baseDir=os.getcwd())
    coords = ["Q20","Q30","pairing"]
    uniqueCoords = [np.unique(dsets[coord]) for coord in coords]
    desiredShape = np.array([len(c) for c in uniqueCoords],dtype=int)
    
    reshapedData = {key:dsets[key].reshape(desiredShape) for key in dsets.keys()}
    cmesh = tuple([reshapedData[coord] for coord in coords])
    zz = reshapedData["E_HFB"]
    minInds = Utilities.find_local_minimum(zz)
    allowedInds = tuple(np.array([inds[zz[minInds]!=-1760] for inds in minInds]))
    
    gsInds = Utilities.extract_gs_inds(allowedInds,cmesh,zz)
    eGS = zz[gsInds]
    zz = zz - eGS
    
    allContours = Utilities.find_approximate_contours(cmesh,zz)
    print(allContours[-1])
    
    fig, ax = plt.subplots()
    cf = ax.contourf(cmesh[0][:,:,-1],cmesh[1][:,:,-1],zz[:,:,-1])
    c = ax.contour(cmesh[0][:,:,-1],cmesh[1][:,:,-1],zz[:,:,-1],levels=[0],colors=["white"])
    print(np.array(c.allsegs).flatten())
    ax.scatter(np.array(c.allsegs).flatten())
    plt.colorbar(cf,ax=ax)
    
    return None

def fire_test():
    #Testing different optimization routines
    startTime = time.time()
    
    fDir = "Daniels_Code/"
    fName = "252U_PES.h5"
    dsets, attrs = FileIO.read_from_h5(fName,fDir)
    
    #Only getting the unique values
    q20Vals = dsets["Q20"][:,0]
    q30Vals = dsets["Q30"][0]
    
    custInterp2d = CustomInterp2d(q20Vals,q30Vals,dsets["PES"].T,kind="quintic")
    potential = Utilities.aux_pot(custInterp2d.potential,0)
    
    interpArgsDict = custInterp2d.kwargs
    interpArgsDict["function"] = custInterp2d.__str__()
    
    nPts = 22
    
    initialPoints = np.array((np.linspace(27,185,nPts),np.linspace(0,15,nPts)))
    initialEnegs = potential(initialPoints)
    constraintEneg = initialEnegs[0]
    
    print("Constraining to energy %.3f" % constraintEneg)
    
    k = 10
    kappa = 20
    lneb = LineIntegralNeb(potential,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    
    maxIters = 2500
    minObj = MinimizationAlgorithms(lneb)
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters)
        
    allPts2, allVelocities2, allForces2, actions2 = \
        minObj.verlet_minimization_with_fire(maxIters=maxIters)
        
    allPts3, allForces3, actions3 = \
        minObj.verlet_minimization(maxIters=maxIters)
    
    """   Plotting   """
    fig, ax = plt.subplots()
    ax.plot(actions3,label="Standard Verlet",color="purple")
    ax.plot(actions,label="Velocity-Projected Verlet",color="blue")
    ax.plot(actions2,label="FIRE Algorithm",color="orange")
    ax.set(xlabel="Iteration",ylabel="Action")
    ax.legend()
    fig.savefig("Runs/"+str(int(startTime))+"_Action.pdf")
    
    cbarRange = (-5,30)
    fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],dsets["PES"])
    ax.contour(dsets["Q20"],dsets["Q30"],dsets["PES"],levels=[constraintEneg],\
                colors=["black"])
    ax.plot(allPts[0,0],allPts[0,1],marker=".",label="Initial Path",color="red")
    ax.plot(allPts3[-1,0],allPts3[-1,1],marker=".",label="Standard Verlet",color="purple")
    ax.plot(allPts[-1,0],allPts[-1,1],marker=".",label="Velocity-Projected Verlet",color="blue")
    ax.plot(allPts2[-1,0],allPts2[-1,1],marker=".",label="FIRE Algorithm",color="orange")
    
    ax.legend(loc="upper left")
    ax.set_xlim(min(q20Vals),max(q20Vals))
    ax.set_ylim(min(q30Vals),max(q30Vals))
    ax.set(title=r"${}^{252}$U PES")
        
    fig.savefig("Runs/"+str(int(startTime))+".pdf")
    
    return None

def test_for_eric():
    #Eric is seeing odd behavior where the point gets stuck
    startTime = time.time()
    
    fDir = "Daniels_Code/"
    fName = "252U_PES.h5"
    dsets, attrs = FileIO.read_from_h5(fName,fDir)
    
    #Only getting the unique values
    q20Vals = dsets["Q20"][:,0]
    q30Vals = dsets["Q30"][0]
    
    custInterp2d = CustomInterp2d(q20Vals,q30Vals,dsets["PES"].T,kind="quintic")
    potential = Utilities.aux_pot(custInterp2d.potential,dsets["PES"].min())
    
    nPts = 52
    
    initialPoints = np.array((np.linspace(25.95,265.96,nPts),np.linspace(0.96,25.31,nPts)))
    actualInitialEneg = custInterp2d.potential(initialPoints)[0]
    initialEnegs = potential(initialPoints)
    constraintEneg = initialEnegs[0]
    
    print("Constraining to energy %.3f" % constraintEneg)
    
    k = 10
    kappa = 20
    lneb = LineIntegralNeb(potential,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    
    maxIters = 1000
    minObj = MinimizationAlgorithms(lneb)
    # allPts, allVelocities, allForces, actions = \
    #     minObj.verlet_minimization_v2(maxIters=maxIters)
        
    allPts2, allVelocities2, allForces2, actions2 = \
        minObj.verlet_minimization_with_fire(maxIters=10,dtMin=0.05,dtMax=1)
    print(len(allPts2))
    
    """   Plotting   """
    fig, ax = plt.subplots()
    # ax.plot(actions,label="Velocity-Projected Verlet",color="blue")
    ax.plot(actions2,label="FIRE",color="orange")
    ax.set(xlabel="Iteration",ylabel="Action")
    ax.legend()
    fig.savefig("Runs/"+str(int(startTime))+"_Action.pdf")
    
    cbarRange = (-5,30)
    fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],dsets["PES"])
    ax.contour(dsets["Q20"],dsets["Q30"],dsets["PES"],levels=[actualInitialEneg],\
                colors=["black"])
    # ax.plot(allPts[0,0],allPts[0,1],marker=".",label="Initial Path",color="red")
    # ax.plot(allPts[-1,0],allPts[-1,1],marker=".",label="Velocity-Projected Verlet",color="blue")
    # ax.plot(allPts2[-1,0],allPts2[-1,1],marker=".",label="FIRE",color="orange")
    for (ptIter,pt) in enumerate(allPts2[:]):
        ax.plot(pt[0],pt[1],marker=".",label=str(ptIter))
    
    ax.legend(loc="upper left")
    ax.set_xlim(min(q20Vals),max(q20Vals))
    ax.set_ylim(min(q30Vals),max(q30Vals))
    ax.set(title=r"${}^{252}$U PES")
        
    # fig.savefig("Runs/"+str(int(startTime))+".pdf")
    
    # print(potential(allPts[-1]))
    
    return None

def interp_mode_test():
    #Grid size for 252U
    nPtsCoarseGrid = (351,151)
    nPtsFineGrid = (501,501)
    
    ptRange = [(0,4),(-2,4)]
    
    xCoarse, yCoarse = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                    num=nPtsCoarseGrid[cIter]) for\
                        cIter in range(len(nPtsCoarseGrid))]
    xDense, yDense = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                  num=nPtsFineGrid[cIter]) for\
                      cIter in range(len(nPtsFineGrid))]
        
    coarseMesh = np.meshgrid(xCoarse,yCoarse)
    zCoarse = LepsPot(initialGrid=coarseMesh).potential(coarseMesh)
    
    denseMesh = np.meshgrid(xDense,yDense)
    zDense = LepsPot(initialGrid=denseMesh).potential(denseMesh)
    
    splineInterps = ["linear","cubic","quintic"]
    interpObjs = {}
    transposeInterpObjs = {}
    interpTimes = {}
    for interpMode in splineInterps:
        t0 = time.time()
        interpObjs[interpMode] = CustomInterp2d(xCoarse,yCoarse,zCoarse,kind=interpMode)
        transposeInterpObjs[interpMode] = CustomInterp2d(yCoarse,xCoarse,zCoarse.T,kind=interpMode)
        t1 = time.time()
        interpTimes[interpMode] = t1 - t0
        
    densePreds = {}
    transposeDensePreds = {}
    predictTimes = {}
    for interpMode in splineInterps:
        t0 = time.time()
        densePreds[interpMode] = interpObjs[interpMode].potential(denseMesh)
        transposeDensePreds[interpMode] = \
            transposeInterpObjs[interpMode].potential(denseMesh[1],denseMesh[0])
        t1 = time.time()
        predictTimes[interpMode] = t1 - t0
        
    clipRange = (-5,-1)
    nLevels = 5
    fig, ax = plt.subplots(nrows=2,ncols=len(splineInterps),\
                           figsize=(4*len(splineInterps),8))
    ax = np.array(ax).reshape((2,len(splineInterps)))
    for (modeIter, mode) in enumerate(splineInterps):
        plotDat = np.log10(np.abs((densePreds[mode]-zDense))).clip(clipRange)
        cf = ax[0,modeIter].contourf(denseMesh[0],denseMesh[1],plotDat,\
                                     levels=np.linspace(clipRange[0],clipRange[1],nLevels))
        
        plotDat = np.log10(np.abs((transposeDensePreds[mode]-zDense))).clip(clipRange)
        cf = ax[1,modeIter].contourf(denseMesh[1],denseMesh[0],plotDat,\
                                    levels=np.linspace(clipRange[0],clipRange[1],nLevels))
            
        ax[0,modeIter].set(xlabel="rAB",ylabel="x",title=mode.capitalize()+" Spline")
        ax[1,modeIter].set(xlabel="x",ylabel="rAB")
        
        ax[0,modeIter].set(xticks=np.arange(5),yticks=np.arange(-2,5))
        ax[1,modeIter].set(yticks=np.arange(5),xticks=np.arange(-2,5))
    
    plt.colorbar(cf,ax=ax[0,-1],label=r"Log${}_{10} | E_{Interp}-E_{Exact}|$")
    
    fig.savefig("LEPs_PES_Diff.pdf",bbox_inches="tight")
    
    # absMeanDiffs = np.array([np.mean(np.abs(densePreds[interpMode] - \
    #                                           zDense)) for interpMode in splineInterps])
    # meanFig, meanAx = plt.subplots()
    # meanAx.scatter(np.arange(len(absMeanDiffs)),np.log10(absMeanDiffs))
    # meanAx.set_xticks(np.arange(len(absMeanDiffs)))
    # meanAx.set_xticklabels([s.capitalize() for s in splineInterps],rotation=45)
    # meanAx.set(xlabel="Spline Mode",ylabel=r"log${}_{10}\langle | E_{Interp}-E_{Exact}|\rangle$")
    # meanAx.set(title="Mean Energy Difference")
    # meanFig.savefig("Mean_PES_Diff.pdf")
    
    # timeFig, timeAx = plt.subplots()
    # timeAx.scatter(np.arange(len(absMeanDiffs)),interpTimes.values(),label="Interpolating")
    # timeAx.scatter(np.arange(len(absMeanDiffs)),predictTimes.values(),label="Predicting")
    # timeAx.set_xticks(np.arange(len(absMeanDiffs)))
    # timeAx.set_xticklabels([s.capitalize() for s in splineInterps],rotation=45)
    # timeAx.legend()
    # timeAx.set(xlabel="Spline Mode",ylabel="Time (s)",title="Total Run Times")
    # timeFig.savefig("Interpolation_Time.pdf")
    
    return None

def gp_test():
    nPtsCoarseGrid = (351,151)
    nPtsFineGrid = (501,501)
    
    ptRange = [(0,4),(-2,3.8)]
    
    xCoarse, yCoarse = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                    num=nPtsCoarseGrid[cIter]) for\
                        cIter in range(len(nPtsCoarseGrid))]
    xDense, yDense = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                  num=nPtsFineGrid[cIter]) for\
                      cIter in range(len(nPtsFineGrid))]
        
    coarseMesh = np.meshgrid(xCoarse,yCoarse)
    zCoarse = LepsPot(initialGrid=coarseMesh).potential(coarseMesh)
    
    denseMesh = np.meshgrid(xDense,yDense)
    zDense = LepsPot(initialGrid=denseMesh).potential(denseMesh)
    
    predKWargs = {"nNeighbors":10}
    gpKWargs = {"kernel":gp.kernels.RBF(length_scale=[0.5,0.25]),"n_restarts_optimizer":20}
    locGP = LocalGPRegressor(xCoarse,yCoarse,gridPot=zCoarse,\
                             predKWargs=predKWargs,gpKWargs=gpKWargs)
    
    testingCutoff = 10#denseMesh[0].shape[0]
    denseCutTuple = tuple([d[:testingCutoff,:testingCutoff] for d in denseMesh])
    t0 = time.time()
    gpPred = locGP.potential(denseCutTuple)
    t1 = time.time()
    print("Elapsed time: %.3f" % (t1 - t0))
    
    f1, a1 = plt.subplots(ncols=2,figsize=(8,4))
    cf = a1[0].contourf(denseCutTuple,zDense[:testingCutoff,:testingCutoff])
    plt.colorbar(cf,ax=a1[0])
    cf = a1[1].contourf(denseCutTuple,gpPred)
    plt.colorbar(cf,ax=a1[1])
    
    Utilities.standard_pes(xDense[:testingCutoff],yDense[:testingCutoff],\
                           zDense[:testingCutoff,:testingCutoff]-gpPred,clipRange=None)
    
    return None

def sylvester_otl_test():
    # t0 = time.time()
    
    sp = SylvesterPot()
    zz = sp.potential(sp.initialGrid)
    # fig, ax = Utilities.standard_pes(sp.initialGrid,zz,clipRange=(-0.1,10))
    
    initialPts = np.array([np.linspace(c[sp.minInds][0],c[sp.minInds][1],num=22) for\
                           c in sp.initialGrid])
    # ax.plot(initialPts,ls="-",marker=".",color="k")
    eConstraint = np.max(zz[sp.minInds])
    
    lneb = LineIntegralNeb(sp.potential,Utilities.const_mass(),initialPts,10,1,eConstraint,\
                           loggingLevel="output")
    minObj = MinimizationAlgorithms(lneb,loggingLevel="output")
    maxIters = 2500
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters,tStep=0.1)
        
    fName = "Daniels_Code"
    lneb.write_log(fName+".h5",overwrite=True)
    minObj.write_log(fName+".h5")
        
    FileIO.write_path("Paths/Daniels_Path.txt",allPts[-1])
    # minInd = np.argmin(actions)
    # ax.plot(allPts[0,0],allPts[0,1,:],marker=".",ls="-")
    # ax.plot(allPts[minInd,0],allPts[minInd,1,:],marker=".",ls="-")
    # ax.plot(allPts[-1,0],allPts[-1,1,:],marker=".",ls="-")
        
    # fig.savefig("Runs/Sylvester_PES.pdf")
        
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(actions.shape[0]),actions)
    # ax.set(xlabel="Iteration",ylabel="Action")
    # fig.savefig("Runs/Sylvester_Action.pdf")
    
    # t1 = time.time()
    # print("Run time: %.3f" % (t1-t0))
    
    return None

#Actually important here lol
if __name__ == "__main__":
    # print(os.listdir("../"))
    # sylvester_otl_test()
    # cProfile.run("sylvester_otl_test()",sort="tottime")
    # compare_paths()
    # nd_contour_test()
    # fire_test()
    # plutonium_test()
    # plutonium_endpoint_test()
    # uranium_test()
    # uranium_pyneb_test()
    u232_test()
    # grid_interp_test()
    # print("asdf")
    # gp_test()
    # interp_mode_test()
    # lps = LepsPot()
    # fig, ax = Utilities.standard_pes(lps.initialGrid,lps.potential(lps.initialGrid))
    # ax.set(xlabel=r"$r_{AB}$",ylabel="x",title="LEPs+HO")
    # fig.savefig("LEPS_Potential.pdf")