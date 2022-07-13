import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import utilities
import fileio

import warnings

def cluster_paths_by_endpoints(listOfPaths,dbscanParams={}):
    # warnings.warn("Method still in development")
    
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

def find_most_similar_paths(firstList,secondList):
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
    warnings.warn("Method still in development")
    
    distancesDict = {}
    nearestIndsDict = {}
    for (p1Iter,p1) in enumerate(firstList):
        e1 = p1[-1]
        distancesDict[p1Iter] = np.inf
        for (p2Iter,p2) in enumerate(secondList):
            e2 = p2[-1]
            dist = np.linalg.norm(e2-e1)
            if dist < distancesDict[p1Iter]:
                distancesDict[p1Iter] = dist
                nearestIndsDict[p1Iter] = p2Iter
    
    return nearestIndsDict, distancesDict
    