import h5py
import os

import numpy as np
import datetime

import warnings
import functools
import inspect

def logging_wrapper(func):
    """
    My thoughts are as follows:
        -Every function/method can handle logging on its own. If it wants to
            write to an HDF5 file, or just dump it to stdout, *it* makes those
            decisions.
        -If a function is not set up to handle any logging, so long as it's
            decorated with @logging_wrapper, logging will be handled here,
            *but* it will probably just be a print to stdout.
        -In a function that we explicitly want to log, especially to HDF5 files,
            custom data types should be specified *in the function* (or class/method).
            That way, this wrapper doesn't have to attempt to guess the appropriate
            format of the inputs/outputs of a function.
        -A logging level of 0 indicates no logging is to take place at all. This is
            assumed when no logging level is specified.
        -A logging level of 1 indicates that only func is to be logged, and not
            its sub-functions. NOT IMPLEMENTED CORRECTLY
        -A logging level of 2 indicates that func and its sub-functions are to
            be logged.
            
    TODO: replace "logging" with a dict of logging arguments:
        {"logging":one of (0,1,2),
         "logLowerFuncs":one of (0,1),
         "logFile":where to dump the log}.
    If logging is 1, set logLowerFuncs to 0 *in this wrapper.* Not sure quite
        what to do if logging is 0 (skip, or set logLowerFuncs to 0). If logging
        is 2, set logLowerFuncs to 1.
    Depending on how things are passed around, may have to be careful to reset
        logging dict afterwards...? Not quite sure on this
    

    Parameters
    ----------
    func : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        if "logging" in kwargs:
            logLevel = kwargs["logging"]
        else:
            logLevel = 0
        
        #Maybe this'll help speed things up in the event that we don't want logging?
        #I'm not really sure how complicated inspect.signature is.
        if logLevel == 0:
            funcOutputs = func(*args,**kwargs)
        else:
            sig = inspect.signature(func)
            for arg in ["logging","logFile"]:
                if (arg not in sig.parameters) and (arg in kwargs):
                    del kwargs[arg]
                    
            funcOutputs = func(*args,**kwargs)
            
            #Prints to stdout if a log is requested, and the logging is not handled
            #in func
            #TODO: integrate with logging package
            if (logLevel in [1,2]) and ("logging" not in kwargs):
                print(75*"=")
                print(datetime.datetime.now().isoformat())
                print(func.__name__)
                print(75*"*")
                print(args)
                print(75*"*")
                print(kwargs)
                print(75*"+")
                print(funcOutputs)
        
        return funcOutputs
    return wrapper

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
    
