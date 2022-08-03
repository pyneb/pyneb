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

def find_most_similar_paths(firstList,secondList,removeDuplicates=False):
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
    # warnings.warn("Method still in development")
    
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
