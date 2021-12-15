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

def get_unique_inds(arr):    
    #From https://stackoverflow.com/a/30003565
    idx_sort = np.argsort(arr)
    sorted_records_array = arr[idx_sort]
    
    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    
    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])
    
    #filter them with respect to their size, keeping only items occurring more than once
    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)
    
    return list(res)

def read_potential():
    fileDir = os.path.expanduser("~/Research/ActionMinimization/PES/")
    
    dsetsToGet = ["Q20","Q30","PES","B2020","B2030","B3030"]
    dsetsDict = {}
    
    h5File = h5py.File(fileDir+"232U.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    uniqueCoords = [np.unique(dsetsDict[key]) for key in coordNms]
    pesShape = [len(c) for c in uniqueCoords]
    dsetsDict = {key:dsetsDict[key].reshape(pesShape).T for key in dsetsDict.keys()}
    
    return dsetsDict

coordNms = ["Q20","Q30"]

pathsDir = os.path.expanduser("~/Research/ActionMinimization/Paths/232U/")
pathFiles = ['Eric_MEP.txt','232U_NDLinear_Mass_None.txt',\
             'Eric_232U_LAP_Mass_True_path.txt',\
             '232U_MAP.txt','232U_MAP_WMP.txt',\
             '232U_PathPablo.txt','232U_PathPablo_MassParams.txt']

pathsDict = {}
for f in pathFiles:
    try:
        pathDf = pd.read_csv(pathsDir+f,comment="#",header=None)
        pathsDict[f] = np.array(pathDf,dtype=float)
    except ValueError:
        pathDf = pd.read_csv(pathsDir+f,comment="#")
        pathsDict[f] = np.array(pathDf,dtype=float)

pesDict = read_potential()

def make_fig():
    fig, ax = plt.subplots()
    # nLevels = 15
    cf = ax.contourf(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=45,\
                     cmap="Spectral_r",extend="both")
    cs = ax.contour(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=10,\
                    linestyles="solid",colors="gray")
    ax.clabel(cs,inline=1,fontsize=5,colors="black")
    
    colorsDict = {'232U_NDLinear_Mass_None.txt':"blue",\
                 'Eric_232U_LAP_Mass_True_path.txt':"blue",\
                 '232U_MAP.txt':"orange",'232U_MAP_WMP.txt':"orange",\
                 '232U_PathPablo.txt':"purple",'232U_PathPablo_MassParams.txt':'purple',\
                 'Eric_MEP.txt':"green"}
    noMassStyle = "solid"
    massStyle = "dotted"
    stylesDict = {'232U_NDLinear_Mass_None.txt':noMassStyle,\
                 'Eric_232U_LAP_Mass_True_path.txt':massStyle,\
                 '232U_MAP.txt':noMassStyle,'232U_MAP_WMP.txt':massStyle,\
                 '232U_PathPablo.txt':noMassStyle,'Eric_MEP.txt':noMassStyle,\
                 '232U_PathPablo_MassParams.txt':massStyle}
        
    for (key,p) in pathsDict.items():
        ax.plot(*p.T,color=colorsDict[key],ls=stylesDict[key])
    
    ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    fig.colorbar(cf)#,label="Energy (MeV)")
    fig.savefig("232U.pdf",bbox_inches="tight")
    
    return None

def make_action_tables():
    uniqueCoords = [np.unique(pesDict[key]) for key in coordNms]
    pes_interp = NDInterpWithBoundary(uniqueCoords,pesDict["PES"])
    massInterpDict = {key:NDInterpWithBoundary(uniqueCoords,pesDict[key]) for\
                      key in ["B2020","B2030","B3030"]}
    mass_interp = mass_funcs_to_array_func(massInterpDict,["20","30"])
    
    
    interpPathsDict = {}
    for key in pathFiles:
        interpPathsDict[key] = InterpolatedPath(pathsDict[key])
        
    useMassDict = {'232U_NDLinear_Mass_None.txt':False,\
                   'Eric_232U_LAP_Mass_True_path.txt':True,\
                   '232U_MAP.txt':False,'232U_MAP_WMP.txt':True,\
                   '232U_PathPablo.txt':False,'Eric_MEP.txt':False,\
                   '232U_PathPablo_MassParams.txt':True}
        
    actsDict = {}
    for key in pathFiles:
        if useMassDict[key]:
            actsDict[key] = interpPathsDict[key].compute_along_path(TargetFunctions.action,500,\
                                                                    tfArgs=[pes_interp],\
                                                                    tfKWargs={"masses":mass_interp})[1][0]
        else:
            actsDict[key] = interpPathsDict[key].compute_along_path(TargetFunctions.action,500,\
                                                                    tfArgs=[pes_interp])[1][0]
    
    print(actsDict)
    return None

make_action_tables()
# make_fig()