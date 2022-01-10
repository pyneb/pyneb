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
pathFiles = ['Eric_MEP.txt','Eric_232U_LAP_Mass_False_path.txt',\
             '232U_LAP_WMP_NEB.txt',\
             '232U_MAP.txt','232U_MAP_WMP.txt',\
             '232U_PathPablo.txt','232U_PathPablo_MassParams.txt',\
             'djk_path_no_mass.txt','djk_path_mass.txt']

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
    
    colorsDict = {'Eric_232U_LAP_Mass_False_path.txt':"blue",\
                 '232U_LAP_WMP_NEB.txt':"blue",\
                 '232U_MAP.txt':"orange",'232U_MAP_WMP.txt':"orange",\
                 '232U_PathPablo.txt':"purple",'232U_PathPablo_MassParams.txt':'purple',\
                 'Eric_MEP.txt':"green",\
                 'djk_path_no_mass.txt':'black','djk_path_mass.txt':'black'}
    noMassStyle = "solid"
    massStyle = "dotted"
    stylesDict = {'Eric_232U_LAP_Mass_False_path.txt':noMassStyle,\
                 '232U_LAP_WMP_NEB.txt':massStyle,\
                 '232U_MAP.txt':noMassStyle,'232U_MAP_WMP.txt':massStyle,\
                 '232U_PathPablo.txt':noMassStyle,'Eric_MEP.txt':noMassStyle,\
                 '232U_PathPablo_MassParams.txt':massStyle,\
                 'djk_path_no_mass.txt':noMassStyle,'djk_path_mass.txt':massStyle}
        
    for (key,p) in pathsDict.items():
        ax.plot(*p.T,color=colorsDict[key],ls=stylesDict[key])
    
    ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    fig.colorbar(cf)#,label="Energy (MeV)")
    fig.savefig("232U.pdf",bbox_inches="tight")
    
    return None

def make_separate_figs():
    colorsDict = {"neb_lap":"blue","dpm":"orange","el":"purple","neb_mep":"green",\
                  "dijkstra":"black"}
    stylesDict = {"no_mass":"solid","mass":"solid"}
    
    noMassFig, noMassAx = plt.subplots()
    massFig, massAx = plt.subplots()
    # nLevels = 15
    for ax in [massAx,noMassAx]:
        cf = ax.contourf(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=45,\
                         cmap="Spectral_r",extend="both")
        cs = ax.contour(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=10,\
                        linestyles="solid",colors="gray")
        ax.clabel(cs,inline=1,fontsize=5,colors="black")
    
    typesDict = {'Eric_232U_LAP_Mass_False_path.txt':["no_mass","neb_lap"],\
                 '232U_LAP_WMP_NEB.txt':["mass","neb_lap"],\
                 '232U_MAP.txt':["no_mass","dpm"],'232U_MAP_WMP.txt':["mass","dpm"],\
                 '232U_PathPablo.txt':["no_mass","el"],'232U_PathPablo_MassParams.txt':["mass","el"],\
                 'Eric_MEP.txt':["no_mass","neb_mep"],'djk_path_no_mass.txt':["no_mass","dijkstra"],\
                 'djk_path_mass.txt':["mass","dijkstra"]}
        
    for (key,p) in pathsDict.items():
        if typesDict[key][0] == "no_mass":
            noMassAx.plot(*p.T,color=colorsDict[typesDict[key][1]],\
                          ls=stylesDict["no_mass"])
        else:
            massAx.plot(*p.T,color=colorsDict[typesDict[key][1]],\
                        ls=stylesDict["mass"])
                
    
    massAx.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    massFig.colorbar(cf)#,label="Energy (MeV)")
    massFig.savefig("232U_mass.pdf",bbox_inches="tight")
    
    noMassAx.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    noMassFig.colorbar(cf)#,label="Energy (MeV)")
    noMassFig.savefig("232U_no_mass.pdf",bbox_inches="tight")
    
    return None

def bw_styles_fig():
    markerDict = {"neb_lap":None,"dpm":None,"el":None,"neb_mep":"o"}
    stylesDict = {"neb_lap":"solid","dpm":"dashed","el":"dotted","neb_mep":None}
    markEvery = 3
    color = "black"
    lineWidth = 0.5
    markerSize = 2
    
    noMassFig, noMassAx = plt.subplots()
    massFig, massAx = plt.subplots()
    # nLevels = 15
    for ax in [massAx,noMassAx]:
        cf = ax.contourf(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=45,\
                         cmap="Spectral_r",extend="both")
        cs = ax.contour(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=10,\
                        linestyles="solid",colors="gray")
        ax.clabel(cs,inline=1,fontsize=5,colors="black")
    
    typesDict = {'Eric_232U_LAP_Mass_False_path.txt':["no_mass","neb_lap"],\
                 '232U_LAP_WMP_NEB.txt':["mass","neb_lap"],\
                 '232U_MAP.txt':["no_mass","dpm"],'232U_MAP_WMP.txt':["mass","dpm"],\
                 '232U_PathPablo.txt':["no_mass","el"],'232U_PathPablo_MassParams.txt':["mass","el"],\
                 'Eric_MEP.txt':["no_mass","neb_mep"]}
        
    for (key,p) in pathsDict.items():
        if typesDict[key][0] == "no_mass":
            ax = noMassAx
        else:
            ax = massAx
        ax.plot(*p.T,color=color,marker=markerDict[typesDict[key][1]],\
                linestyle=stylesDict[typesDict[key][1]],markevery=markEvery,\
                linewidth=lineWidth,markersize=markerSize)
    
    massAx.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    massFig.colorbar(cf,ax=massAx)#,label="Energy (MeV)")
    massFig.savefig("232U_mass_styled.pdf",bbox_inches="tight")
    
    noMassAx.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    noMassFig.colorbar(cf,ax=noMassAx)#,label="Energy (MeV)")
    noMassFig.savefig("232U_no_mass_styled.pdf",bbox_inches="tight")
    
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
        
    useMassDict = {'Eric_232U_LAP_Mass_False_path.txt':False,\
                   '232U_LAP_WMP_NEB.txt':True,\
                   '232U_MAP.txt':False,'232U_MAP_WMP.txt':True,\
                   '232U_PathPablo.txt':False,'Eric_MEP.txt':False,\
                   '232U_PathPablo_MassParams.txt':True,\
                   'djk_path_no_mass.txt':False,'djk_path_mass.txt':True}
        
    actsDict = {}
    for key in pathFiles:
        # if useMassDict[key]:
        actsDict[key] = interpPathsDict[key].compute_along_path(TargetFunctions.action,500,\
                                                                tfArgs=[pes_interp],\
                                                                tfKWargs={"masses":mass_interp})[1][0]
        # else:
        #     actsDict[key] = interpPathsDict[key].compute_along_path(TargetFunctions.action,500,\
        #                                                             tfArgs=[pes_interp])[1][0]
    
    r1Keys = ["Eric_MEP.txt","Eric_232U_LAP_Mass_False_path.txt","232U_MAP.txt","232U_PathPablo.txt",\
              "djk_path_no_mass.txt"]
    r2Keys = ["232U_LAP_WMP_NEB.txt","232U_MAP_WMP.txt","232U_PathPablo_MassParams.txt",\
              "djk_path_mass.txt"]
        
    #Have to format here and in tabulate, since it'll treat different columns differently
    r1 = [r'${}^{232}$U']+[format(actsDict[key],".1f") for key in r1Keys]
    r2 = [r'${}^{232}$U, WI',"-"]+[format(actsDict[key],".1f") for key in r2Keys]
    
    headers = ["NEB-MEP","NEB-LAP","DPM","EL","Dijkstra"]
    
    print(tabulate([r1,r2], headers=headers, tablefmt='latex_raw',floatfmt=".1f"))
    
    # print(actsDict)
    return None

np.seterr(invalid="raise")
make_action_tables()
make_fig()
make_separate_figs()
# bw_styles_fig()