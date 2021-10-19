import h5py
import os

import numpy as np
import datetime

import warnings
import functools
import inspect

class DijkstraLogger:
    def __init__(self,djkInst,logLevel=1):
        self.logLevel = logLevel
        self.initTime = datetime.datetime.now().isoformat()
        self.djkInst = djkInst
        os.makedirs("logs",exist_ok=True)
        
        if self.logLevel == 1:
            self.fileName = "logs/"+self.initTime+".djk"
            
            h5File = h5py.File(self.fileName,"w")
            
            h5File.attrs.create("initialInds",np.array(self.djkInst.initialInds))
            h5File.attrs.create("initialPoint",np.array(self.djkInst.initialPoint))
            h5File.attrs.create("target_func",self.djkInst.target_func.__qualname__)
            
            #TODO: allow for naming the coordinates
            h5File.create_group("uniqueCoords")
            for (cIter, coord) in enumerate(self.djkInst.uniqueCoords):
                h5File["uniqueCoords"].create_dataset("coord_"+str(cIter),\
                                                      data=np.array(coord))
            h5File.create_dataset("potArr",data=self.djkInst.potArr)
            if djkInst.trimVals[0] is not None:
                h5File["potArr"].attrs.create("minTrim",data=self.djkInst.trimVals[0])
            if djkInst.trimVals[1] is not None:
                h5File["potArr"].attrs.create("maxTrim",data=self.djkInst.trimVals[1])
                
            h5File.create_dataset("inertArr",data=self.djkInst.inertArr)
            
            h5File.create_dataset("endpointIndices",data=np.array(self.djkInst.endpointIndices))
            h5File.create_dataset("allowedEndpoints",data=self.djkInst.allowedEndpoints)
            
            h5File.close()
        
    def log(self,variables,variableNames):
        if self.logLevel == 0:
            return None
        elif self.logLevel == 1:
            self._write_level_1(variables,variableNames)
        
        return None
            
    def _write_level_1(self,variables,variableNames):
        """
        Want to store:
            -run time

        Returns
        -------
        None.

        """
        h5File = h5py.File(self.fileName,"a")
        for (var, nm) in zip(variables,variableNames):
            #Unfortunately this doesn't seem to be abstractable, but perhaps I
            #don't need it to be
            if nm == "tentativeDistance":
                #HDF5 files can store np.inf in an array just fine
                dtype = np.dtype({"names":["data","mask"],"formats":[float,bool]})
                arr = np.zeros(var.shape,dtype=dtype)
                arr["data"] = var.data
                arr["mask"] = var.mask
                h5File.create_dataset(nm,data=arr)
            elif nm == "neighborsVisitDict":
                dtype = np.dtype([("key",int,(self.djkInst.nDims,)),\
                                  ("val",int,(self.djkInst.nDims,))])
                arr = np.zeros(len(var.keys()),dtype=dtype)
                for (keyIter,key) in enumerate(var.keys()):
                    arr[keyIter]["key"] = key
                    arr[keyIter]["val"] = var[key]
                h5File.create_dataset(nm,data=arr)
            elif nm == "allPathsIndsDict":
                maxSize = np.max([len(path) for path in var.values()])
                dtype = np.dtype([("finalInd",int,(self.djkInst.nDims,)),\
                                  ("nPts",int),\
                                  ("pathInds",int,(maxSize,self.djkInst.nDims))])
                #Only the first nPts points are valid
                arr = np.zeros(len(var.keys()),dtype=dtype)
                for (keyIter,key) in enumerate(var.keys()):
                    arr["finalInd"][keyIter] = np.array(key)
                    path = np.array(var[key])
                    nPts = path.shape[0]
                    arr["nPts"][keyIter] = nPts
                    arr["pathInds"][keyIter,:nPts] = path
                h5File.create_dataset(nm,data=arr)
            elif nm == "pathArrDict":
                maxSize = np.max([path.shape[0] for path in var.values()])
                dtype = np.dtype([("finalPoint",float,(self.djkInst.nDims,)),\
                                  ("nPts",int),\
                                  ("path",float,(maxSize,self.djkInst.nDims))])
                #Only the first nPts points are valid
                arr = np.zeros(len(var.keys()),dtype=dtype)
                for (keyIter,key) in enumerate(var.keys()):
                    arr["finalPoint"][keyIter] = np.array(key)
                    path = np.array(var[key])
                    nPts = path.shape[0]
                    arr["nPts"][keyIter] = nPts
                    arr["path"][keyIter,:nPts] = path
                h5File.create_dataset(nm,data=arr)
            elif nm == "endptOut":
                h5File.attrs.create("minimalEndpt",np.array(var))
            elif nm == "endpointIndsList":
                if len(var) > 0:
                    h5File.create_dataset("unvisitedEndpoints",data=np.array(var))
            elif nm == "runTime":
                h5File.attrs.create(nm,var)
            else:
                warnings.warn("Variable "+nm+" not logged to HDF5 file")
        
        h5File.close()
        return None
    
