import os
import sys

from tabulate import tabulate
from texttable import Texttable
import latextable

import pandas as pd

pyNebDir = os.path.expanduser("~/Research/ActionMinimization/py_neb/")
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *

plt.style.use('science')

#Use np.unique(arr,axis=0)
# def get_unique_inds(arr):    
#     #From https://stackoverflow.com/a/30003565
#     idx_sort = np.argsort(arr)
#     sorted_records_array = arr[idx_sort]
    
#     # returns the unique values, the index of the first occurrence of a value, and the count for each element
#     vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    
#     # splits the indices into separate arrays
#     res = np.split(idx_sort, idx_start[1:])
    
#     #filter them with respect to their size, keeping only items occurring more than once
#     vals = vals[count > 1]
#     res = filter(lambda x: x.size > 1, res)
    
#     return list(res)

def read_potential():
    fileDir = os.path.expanduser("~/Research/ActionMinimization/PES/")
    
    dsetsToGet = ["Q20","Q30","pairing","E_HFB","B2020","B2030","B3030","B20pair","B30pair","Bpairpair"]
    dsetsDict = {}
    
    h5File = h5py.File(fileDir+"240Pu.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    uniqueCoords = [np.unique(dsetsDict[key]) for key in coordNms]
    pesShape = [len(c) for c in uniqueCoords]
    dsetsDict = {key:dsetsDict[key].reshape(pesShape) for key in dsetsDict.keys()}
    
    return dsetsDict

coordNms = ["Q20","Q30","pairing"]

pathsDir = os.path.expanduser("~/Research/ActionMinimization/Paths/240Pu/")
pathFiles = [f for f in os.listdir(pathsDir) if f.endswith("txt")]

pathsDict = {}
for f in pathFiles:
    try:
        pathDf = pd.read_csv(pathsDir+f,comment="#",header=None)
        pathsDict[f] = np.array(pathDf,dtype=float)
    except ValueError:
        pathDf = pd.read_csv(pathsDir+f,comment="#")
        pathsDict[f] = np.array(pathDf,dtype=float)
        
pesDict = read_potential()


def make_action_tables():
    uniqueCoords = [np.unique(pesDict[key]) for key in coordNms]
        
    gsInds = SurfaceUtils.find_local_minimum(pesDict["E_HFB"],searchPerc=3*[0.25])
    E_gs = pesDict["E_HFB"][gsInds]
    zz = pesDict["E_HFB"] - E_gs
    # print("E_gs: %.4f"%E_gs)
    pes_interp = NDInterpWithBoundary(uniqueCoords,zz)
    
    massInterpDict = {key:NDInterpWithBoundary(uniqueCoords,pesDict[key]) for\
                      key in ["B2020","B2030","B3030","B20pair","B30pair","Bpairpair"]}
    mass_interp = mass_funcs_to_array_func(massInterpDict,["20","30","pair"])
    
    for key in pathFiles:
        if pathsDict[key].shape[1] == 2:
            print(key)
            pathsDict[key] = np.hstack((pathsDict[key],np.zeros((pathsDict[key].shape[0],1))))
    
    interpPathsDict = {}
    for key in pathFiles:
        interpPathsDict[key] = InterpolatedPath(pathsDict[key])
                
    actsDict = {}
    for key in pathFiles:
        actsDict[key] = interpPathsDict[key].compute_along_path(TargetFunctions.action,500,\
                                                                tfArgs=[pes_interp],\
                                                                tfKWargs={"masses":mass_interp})[1][0]
    
    for (key,val) in actsDict.items():
        # print(50*"=")
        print(key+", %.3f"%val)
        
    # print(actsDict)
    
    return None

np.seterr(invalid="raise")
make_action_tables()
