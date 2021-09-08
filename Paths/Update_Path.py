import sys, os
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
from math import ceil

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

def read_from_h5(fName):
    datDictOut = {}
    attrDictOut = {}
    
    h5File = h5py.File(fName,"r")
    allDataSets = [key for key in h5_get_keys(h5File) if isinstance(h5File[key],h5py.Dataset)]
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

def read_path(fname,returnHeads=False):
    df = pd.read_csv(fname,sep=",",index_col=None,header=None,comment="#",engine="python")
    
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

def get_all_paths(nuc):
    pathDir = os.getcwd()+"/"+nuc
    
    allFiles = os.listdir(pathDir)
    pathFiles = [f for f in allFiles if f.endswith(".txt") or f.endswith(".csv")]
    pathFiles.sort()
    
    pathDict = {}
    for f in pathFiles:
        label = f.replace(".txt","").replace(".csv","").replace(nuc,"").lstrip("_")
        pathDict[label] = read_path(pathDir+"/"+f)
    
    return pathDict

def standard_pes(xx,yy,zz,clipRange=(-5,30)):
    fig, ax = plt.subplots()
    if clipRange is None:
        clipRange = (zz.min()-0.2,zz.max()+0.2)
    #USE THIS COLORMAP FOR PESs - has minima in blue and maxima in red
    cf = ax.contourf(xx,yy,zz.clip(clipRange[0],clipRange[1]),\
                     cmap="Spectral_r",levels=np.linspace(clipRange[0],clipRange[1],25))
    plt.colorbar(cf,ax=ax)
    return fig, ax

def main(nuc):
    pesDir = os.getcwd()+"/..//PES"
    dsets, attrs = read_from_h5(pesDir+"/"+nuc+".h5")
    
    possibleCoordStrs = ["Q20","Q30","pairing"]
    coordDict = {key:dsets[key] for key in possibleCoordStrs if key in dsets.keys()}
    
    gridShape = [len(np.unique(d)) for d in coordDict.values()]
    cmeshTuple = tuple([c.reshape(gridShape) for c in coordDict.values()])
    zz = dsets["PES"].reshape(gridShape)
    
    paths = get_all_paths(nuc)
    
    if len(coordDict.keys()) == 2:
        fig, ax = standard_pes(*cmeshTuple,zz)
        ax.contour(*cmeshTuple,zz,levels=[0],colors=["black"])
        
        #TODO: include automatic handling of different coordinates
        ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)")
        for key in paths.keys():
            if paths[key].shape[0] > 50:
                ls = None
            else:
                ls = "."
            ax.plot(paths[key][:,0],paths[key][:,1],marker=ls,label=key)
            
        ax.set(title=nuc)
            
        if len(paths.keys()) > 6:
            fig.savefig(os.getcwd()+"/"+nuc+"/"+nuc+"_no_legend.pdf")
            print("Warning: making legend with "+str(len(paths.keys()))+" keys")
            
        ax.legend(loc="upper left")
        fig.savefig(os.getcwd()+"/"+nuc+"/"+nuc+".pdf")
    else:
        sys.exit("Err: requested PES with "+str(len(coordDict.keys()))+\
                 " coordinates; code only set up to handle 2 coordinates")
    
    return None

if __name__ == "__main__":
    nuc = sys.argv[1]
    main(nuc)