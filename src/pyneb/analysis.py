import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.preprocessing import StandardScaler

import utilities
import fileio

import sys, os
import shutil
import subprocess
import h5py
import pandas as pd
import typing

import warnings

def cluster_endpoints(allPaths,pathActions,meanshiftParams={},
                      silenceWarnings=False):
    """
    Clusters paths by their endpoints, using sklearn.cluster.MeanShift,
    and selects the path with the least action for each cluster

    Parameters
    ----------
    allPaths : TYPE
        DESCRIPTION.
    pathActions : TYPE
        DESCRIPTION.
    silenceWarnings : TYPE, optional
        DESCRIPTION. The default is False.

    Yields
    ------
    nCats : TYPE
        DESCRIPTION.
    uniquePaths : TYPE
        DESCRIPTION.
    uniqueInds : TYPE
        DESCRIPTION.
    clustering : TYPE
        DESCRIPTION.

    """
    
    defaultMeanshiftParams = {'bandwidth':5}
    for (key,val) in defaultMeanshiftParams.items():
        if key not in meanshiftParams:
            meanshiftParams[key] = val
        
    pathActions = np.array(pathActions)
    
    endpoints = np.array([p[-1] for p in allPaths])
    clustering = MeanShift(**meanshiftParams).fit(endpoints)
    
    nCats = len(np.unique(clustering.labels_))
    
    uniquePaths = []
    uniqueInds = []
    for i in range(nCats):
        matchInds = np.where(clustering.labels_==i)[0]
        uniqueInds.append(matchInds)
        #Some paths yield the same endpoint, but take longer to get there. Select
        #the minimum here
        actMinMatchInd = matchInds[np.argmin(pathActions[matchInds])]
        uniquePaths.append(allPaths[actMinMatchInd])
        
    return nCats, uniquePaths, uniqueInds, clustering

def cluster_paths_by_endpoints(listOfPaths,dbscanParams={},
                               silenceWarnings=False):
    if not silenceWarnings:
        warnings.warn("Method 'cluster_paths_by_endpoints' still in development")
    warnings.warn("Method 'cluster_path_by_endpoints' will be deprecated in favor of 'cluster_paths_by_endpoints'")
    
    defaultDbscanParams = {"eps":0.3,"min_samples":1}
    for (key,val) in defaultDbscanParams.items():
        if key not in dbscanParams:
            dbscanParams[key] = val
    
    endPoints = np.array([p[-1] for p in listOfPaths])
    
    rescaledPoints = StandardScaler(with_std=False).fit_transform(endPoints)
    db = DBSCAN(**dbscanParams).fit(rescaledPoints)
    
    labels = np.array(db.labels_)
    nClusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    uniquePaths = []
    uniqueInds = []
    for i in range(nClusters):
        idx = np.where(labels==i)[0][0]
        uniqueInds.append(idx)
        uniquePaths.append(listOfPaths[idx])
    
    return uniquePaths, uniqueInds

def find_most_similar_paths(firstList,secondList,removeDuplicates=False,
                            silenceWarnings=False):
    """
    For every path in firstList, finds the path in secondList that is the closest.
    
    Does
    not do anything more sophisticated, like check if the paths 'look like'
    each other, because that's prohibitively complicated (I don't know how to
    do it).
    
    Allows for lists of different lengths.

    Parameters
    ----------
    firstList : TYPE
        DESCRIPTION.
    secondList : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not silenceWarnings:
        warnings.warn("Method 'find_most_similar_paths' still in development")
    
    nearestIndsArr = np.nan*np.ones(len(firstList),dtype=int)
    distancesArr = np.inf*np.ones(len(firstList))
    for (p1Iter,p1) in enumerate(firstList):
        e1 = p1[-1]
        for (p2Iter,p2) in enumerate(secondList):
            e2 = p2[-1]
            dist = np.linalg.norm(e2-e1)
            if dist < distancesArr[p1Iter]:
                distancesArr[p1Iter] = dist
                nearestIndsArr[p1Iter] = p2Iter
    
    if removeDuplicates:
        for uniqueP2Inds in np.unique(nearestIndsArr):
            indsToCheck = np.where(nearestIndsArr==uniqueP2Inds)[0]
            minInd = indsToCheck[0]
            minDist = distancesArr[indsToCheck[0]]
            for i in indsToCheck[1:]:
                d = distancesArr[i]
                if d < minDist:
                    nearestIndsArr[minInd] = np.nan
                    distancesArr[minInd] = np.nan
                    minDist = d
                    minInd = i
                else:
                    nearestIndsArr[i] = np.nan
                    distancesArr[i] = np.nan
    #     nOccurrences = np.zeros(len(set(nearestIndsDict.values())))
    #     for p2Ind in nearestIndsDict.values():
    #         nOccurrences[p2Ind] += 1
    #     print(nOccurrences)
        # for (p1Ind,p2Ind) in nearestIndsDict.items():
            
    return nearestIndsArr, distancesArr

class LAPFilters:
    @staticmethod
    def correct_starting_point(path:np.ndarray[float],gsLoc:np.ndarray[float],
                               tol:float=0.5) -> tuple[bool, np.ndarray[float], str]:
        isValid = True
        msg = 'Success'
        pathOut = path

        if np.any(np.abs(path[0] - gsLoc) > tol):
            isValid = False
            msg = 'Starting location to far from desired point'
            pathOut = None
        return isValid, pathOut, msg
        
    @staticmethod
    def monotonic(path:np.ndarray[float],diffIsStrict:bool=True,
                  coordinateIdx:int=0) -> tuple[bool, np.ndarray[float], str]:
        diffArr = np.diff(path[:,coordinateIdx])
    
        isValid = True
        msg = 'Success'
        pathOut = path

        if diffIsStrict:
            if np.any(diffArr<0):
                isValid = False
                pathOut = None
                msg = 'Non-monotonic in coordinate {:d}'.format(coordinateIdx)
        else:
            badInds = np.where(diffArr<=0)[0]
            if len(badInds) > 0:
                path = path[:badInds[0]]
                if path.shape[0] <= 4: #Basically arbitrary
                    isValid = False
                    pathOut = None
                    msg = 'Insufficient monotonic points in first coordinate'
                else:
                    isValid = True
                    pathOut = path
                    msg = 'Success with non-strict monotonic filter'
                
        return isValid, pathOut, msg
    
    @staticmethod
    def out_of_bounds(path:np.ndarray[float],
                      bounds:np.ndarray[float]) -> tuple[bool, np.ndarray[float], str]:
        """
        Does exactly what it says. path is shape [nPoints,nCoordinates],
        bounds is shape [nCoordinates,2]
        """
        isValid = True
        pathOut = path
        msg = 'Success'
        for coordIter in range(path.shape[1]):
            if (np.any(path[:,coordIter] < bounds[coordIter,0])) or \
                (np.any(path[:,coordIter] > bounds[coordIter,1])):
                isValid = False
                pathOut = []
                msg = 'LAP goes out-of-bounds'
        return isValid, pathOut, msg
    
    @staticmethod
    def energy_within_tolerance(enegOnPath:np.ndarray[float],
                                energyToMatch:float,
                                index:int,energyTol:float=0.1):
        """
        TODO: don't like that this has a different return signature
        than LAPFilters.monotonic
        """
        #Some parameter sets fail for initial guesses, and pull the endpoint off of
        #the OTL
        if (np.abs(enegOnPath[index] - energyToMatch) > energyTol):
            valid = False
            msg = 'Energy at index {:d} greater than enegUpperThresh'.format(index)
        else:
            valid = True
            msg = 'Success'
        
        return valid, msg

def truncate_path_at_given_energy(path:np.ndarray[float],pes_func:callable,
                                  energy:float=0.0,energyTol:tuple[float]=(0.05,0.1),
                                  nSkip:int=100,nInterp:int=500):
    interpPath = utilities.InterpolatedPath(path)
    densePath = np.array(interpPath(np.linspace(0,1,nInterp))).T

    enegOnPath = pes_func(densePath)

    enegInds = np.where(np.abs(enegOnPath - energy)[nSkip:] < energyTol[0])[0]
    
    if len(enegInds) > 0:
        indToTruncateAt = enegInds[0]+nSkip
    else:
        indToTruncateAt = -1
    
    return densePath[:indToTruncateAt], enegOnPath[:indToTruncateAt]

def filter_path(path,pes_func,diffFilter=True,diffIsStrict=True,enegLowerThresh=0.05,
                enegUpperThresh=0.1,nSkip=100,enegFilter=True,silenceWarnings=False,
                minSkipCoord=None):
    """
    Filters a path according to a number of criteria. Currently checks if
    path is monotonic in the first coordinate. It then interpolates the path
    to 500 points, and looks for points on the interpolated path with energy
    near zero (i.e. it intersects the outer turning line multiple times). If
    multiple points, after nSkip of the 500 interpolation points, intersect, it
    truncates the path at that point. If the endpoint is above enegUpperThresh,
    it is deemed invalid.
    

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    pes_func : TYPE
        DESCRIPTION.
    diffFilter : TYPE, optional
        DESCRIPTION. The default is True.
    enegLowerThresh : TYPE, optional
        DESCRIPTION. The default is 0.05.
    enegUpperThresh : TYPE, optional
        DESCRIPTION. The default is 0.1.
    nSkip : TYPE, optional
        DESCRIPTION. The default is 100.
    enegFilter : TYPE, optional
        DESCRIPTION. The default is True.
    silenceWarnings : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    warnings.warn("pyneb.analysis.filter_path is deprecated in favor of LAPFilters",
                  category=DeprecationWarning,stacklevel=2)
    diffArr = np.diff(path[:,0])
    
    if diffFilter:
        if diffIsStrict:
            if np.any(diffArr<0):
                return [], [], 0, False, 'Non-monotonic in first coordinate'
        else:
            badInds = np.where(diffArr<=0)[0]
            if len(badInds) > 0:
                path = path[:badInds[0]]
                if path.shape[0] <= 4: #Basically arbitrary
                    return [], [], 0, False, 'Insufficient monotonic points in first coordinate'
     
    interpPath = utilities.InterpolatedPath(path)
    densePath = np.array(interpPath(np.linspace(0,1,500))).T
    
    enegOnPath = pes_func(densePath)
    
    if enegFilter:
        if minSkipCoord is not None:
            indsToUse = np.where(densePath[:,0]>minSkipCoord)[0]
            if len(indsToUse)>0:
                nSkip = indsToUse[0]
            else:
                nSkip = len(densePath)

        enegInds = np.where(np.abs(enegOnPath)[nSkip:] < enegLowerThresh)[0]
    
        if len(enegInds) > 0:
            indToTruncateAt = enegInds[0]+nSkip
        else:
            indToTruncateAt = -1
        
        #Some parameter sets fail for initial guesses, and pull the endpoint off of
        #the OTL
        if np.abs(enegOnPath[indToTruncateAt]) > enegUpperThresh:
            valid = False
            errorReason = 'Final energy greater than enegUpperThresh'
        else:
            valid = True
            errorReason = None
        
        return densePath, enegOnPath, indToTruncateAt, valid, errorReason
    else:
        return densePath, enegOnPath, -1, True, None
    
def get_relative_probabilities(actions:np.ndarray):
    actMin = np.min(actions)
    relativeProbs = np.exp(-2*(actions-actMin))
    probs = np.exp(-2*actions)
    return probs, relativeProbs
    
def action_is_relevant(actions,thresh=0.01):
    actMin = np.min(actions)
    relativeProbs = np.exp(-2*(actions-actMin))
    sortInds = np.argsort(actions)
    
    cumulativeRelProbs = np.zeros(len(actions))
    for i in range(len(actions)):
        cumulativeRelProbs[i] = np.sum(relativeProbs[sortInds][:i+1])
    percDiff = np.diff(cumulativeRelProbs)/cumulativeRelProbs[-1]
    
    pathIsRelevant = np.zeros(len(actions),dtype=bool)
    pathIsRelevant[sortInds[0]] = True
    for (pIter,p) in enumerate(percDiff):
        if p > thresh:
            pathIsRelevant[sortInds[pIter+1]] = True
            
    nRelevantPaths = np.sum(pathIsRelevant)
            
    return pathIsRelevant, cumulativeRelProbs, actMin, nRelevantPaths

def trim_neb_log(logFile,target_func):
    """
    Computes loss at each iteration using target_func, then deletes all but the
    last iteration for each dataset that is saved on an iteration-by-iteration
    basis (the names are hard-coded in). In this way, one can still track
    convergence (e.g. for archival purposes).
    
    The original file is not deleted.
    
    Notes
    -----
    If the log level only saves data at the final iteration, do NOT run this.

    Parameters
    ----------
    logFile : TYPE
        DESCRIPTION.
    target_func : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    tmpFile = 'tmp.lap'
    newFile = 'trimmed_'+logFile
    
    shutil.copy(logFile,tmpFile)
    
    with h5py.File(tmpFile,'a') as h5File:
        tfEval = target_func(np.array(h5File['points']))
        
        h5File.create_dataset('loss_values',data=tfEval)
        
        for key in ['netForce','perpSpringForce','points','springForce',\
                    'tStep','tangents',]:
            lastVal = h5File[key][-1]
            
            del h5File[key]
            
            h5File.create_dataset(key,data=lastVal)
    
    subprocess.run(['h5repack',tmpFile,newFile])
    
    os.remove(tmpFile)
    
def plot_pes_from_dataframe(pes:pd.DataFrame,
                            plotCol:str,
                            coordCols:typing.List[str]=['Q20','Q30'],
                            cmap='Spectral_r',
                            clipRange=None,
                            maskCol:str=None,
                            maskNegate:bool=False):
    assert len(coordCols) == 2

    uniqueCoords = [np.unique(pes[col].to_numpy()) for col in coordCols]
    shp = [len(u) for u in uniqueCoords]

    plotArr = pes[plotCol].to_numpy().reshape(shp)
    
    if clipRange is None:
        clipRange = (np.nanmin(plotArr),np.nanmax(plotArr))
    
    if maskCol is not None:
        maskArr = pes[maskCol].to_numpy().reshape(shp)
        if maskNegate:
            plotArr = np.ma.masked_array(plotArr,mask=~maskArr)
        else:
            plotArr = np.ma.masked_array(plotArr,mask=maskArr)

    fig, ax = plt.subplots()
    cf = ax.contourf(*uniqueCoords,plotArr.T.clip(*clipRange),cmap=cmap,
                     levels=30)
    plt.colorbar(cf,ax=ax)

    ax.set(xlabel=coordCols[0],ylabel=coordCols[1],
           title=plotCol)
    return fig, ax

def standard_annotations_on_fig(fig,ax,pes,addGSloc=True,
                                addGScontour=True,
                                markInterpolatedPoints=True,
                                coordCols:typing.List[str]=['Q20','Q30']):
    if markInterpolatedPoints:
        ax.scatter(pes[pes['isInterpolated']][coordCols[0]],
                   pes[pes['isInterpolated']][coordCols[1]],
                   marker='x',color='k')
    return fig, ax
