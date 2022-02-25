import os

import sys
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
from texttable import Texttable
import latextable

import pandas as pd

# rootDir = os.getcwd()+"..//.."
# print(rootDir)

pyNebDir = os.path.expanduser("~/ActionMinimization/py_neb/")
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
    fileDir = os.path.expanduser("~/ActionMinimization/PES/")
    
    dsetsToGet = ["Q20","Q30","PES","B2020","B2030","B3030"]
    dsetsDict = {}
    
    h5File = h5py.File(fileDir+"232U-SkMs-NF25.h5","r")
    for dset in dsetsToGet:
        dsetsDict[dset] = np.array(h5File[dset])
    
    h5File.close()
    
    uniqueCoords = [np.unique(dsetsDict[key]) for key in coordNms]
    pesShape = [len(c) for c in uniqueCoords]
    dsetsDict = {key:dsetsDict[key].reshape(pesShape).T for key in dsetsDict.keys()}
    
    return dsetsDict

coordNms = ["Q20","Q30"]

pathsDir = os.path.expanduser("~/ActionMinimization/Paths/232U/")
pathFiles = [f for f in os.listdir(pathsDir) if f.endswith(".txt")]

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
    cf = ax.contourf(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(0.01,30),levels=MaxNLocator(nbins = 45).tick_values(0,25),\
                     cmap="Spectral_r",extend="both")
    cs = ax.contour(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(0.01,30),levels=10,\
                    linestyles="solid",colors="black",linewidths=.5)
    ax.contour(pesDict["Q20"],pesDict["Q30"],pesDict["PES"],levels=[0],colors="white",linewidths=2.5)
    ax.clabel(cs,inline=1,fontsize=5,colors="black")
    
    colorsDict = {"neb-lap":"blue","neb-mep":"green","dpm":"orange",
                  "djk":"black","el":"purple"}
    stylesDict = {"no-mass":"solid","mass":"dotted"}
        
    for (key,p) in pathsDict.items():
        colorKey, styleKey = key.replace(".txt","").split("_")
        ax.plot(*p.T,color=colorsDict[colorKey],ls=stylesDict[styleKey])
    
    fig.colorbar(cf,ax=ax)#,label="Energy (MeV)")
    ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)",xlim=(0,400),ylim=(0,50))
    
    fig.savefig("232U.pdf",bbox_inches="tight")
    
    return None

def make_separate_figs():
    colorsDict = {"neb-lap":"blue","dpm":"orange","el":"purple","neb-mep":"green",\
                  "djk":"black"}
    stylesDict = {"no-mass":"solid","mass":"solid"}
    
    noMassFig, noMassAx = plt.subplots()
    massFig, massAx = plt.subplots()
    # nLevels = 15
    for ax in [massAx,noMassAx]:
        cf = ax.contourf(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=45,\
                         cmap="Spectral_r",extend="both")
        cs = ax.contour(pesDict["Q20"],pesDict["Q30"],pesDict["PES"].clip(-5,30),levels=10,\
                        linestyles="solid",colors="gray")
        ax.clabel(cs,inline=1,fontsize=5,colors="black")
        
    for (key,p) in pathsDict.items():
        colorKey, styleKey = key.replace(".txt","").split("_")
        if styleKey == "no-mass":
            noMassAx.plot(*p.T,color=colorsDict[colorKey],\
                          ls=stylesDict["no-mass"])
        else:
            massAx.plot(*p.T,color=colorsDict[colorKey],\
                          ls=stylesDict["no-mass"])
                
    
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
        
    actsDict = {}
    for key in pathFiles:
        actsDict[key] = interpPathsDict[key].compute_along_path(TargetFunctions.action,500,\
                                                                tfArgs=[pes_interp],\
                                                                tfKWargs={"masses":mass_interp})[1][0]
    
    tableOrder = ["neb-lap","dpm","el","djk"]
    noMassKeys = ["mep-neb.txt"]+[nm+"_no-mass.txt" for nm in tableOrder]
    massKeys = [nm+"_mass.txt" for nm in tableOrder]
        
    #Have to format here and in tabulate, since it'll treat different columns differently
    r1 = [r'${}^{232}$U']
    for key in noMassKeys:
        if key in actsDict.keys():
            r1 += [format(actsDict[key],".1f")]
        else:
            r1 += "-"
    
    r2 = [r'${}^{232}$U, WI',"-"]
    for key in massKeys:
        if key in actsDict.keys():
            r2 += [format(actsDict[key],".1f")]
        else:
            r2 += "-"
    
    # +[format(actsDict[key],".1f") for key in r2Keys]
    # +[format(actsDict[key],".1f") for key in r1Keys]
    
    
    headers = ["NEB-MEP","NEB-LAP","DPM","EL","DJK"]
    
    print(tabulate([r1,r2], headers=headers, tablefmt='latex_raw',floatfmt=".1f"))
    
    # print(actsDict)
    return None

np.seterr(invalid="raise")
make_fig()
make_separate_figs()
make_action_tables()