import numpy as np
from scipy.ndimage import filters, morphology #For minimum finding
import h5py
import sys

"""
CONVENTIONS:
    -Paths should be of shape (nPoints, nDimensions)
    -Functions (e.g. a potential) that take in a single point should assume the
        first index of the array iterates over the points
    -Similarly, functions (e.g. the action) that take in many points should also
        assume the first index iterates over the points
"""

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
        
    if massArr.shape != (nPoints, nDims, nDims):
        sys.exit("Err: massArr "+str(massArr)+" not allowed; see action function")
    
    if not isinstance(potential,np.ndarray):
        potArr = potential(path)
    else:
        potArr = potential
        
    if potArr.shape != (nPoints,):
        sys.exit("Err: potArr "+str(potArr)+" not allowed; see action function")
        
    #Actual calculation
    actOut = 0
    for ptIter in range(1,nPoints):
        coordDiff = path[ptIter] - path[ptIter - 1]
        dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
        actOut += np.sqrt(2*potArr[ptIter]*dist)
    
    return actOut

# class Least_action_path_NEB():
    
# class Minimum_energy_path_NEB():


    