import h5py
import os

import numpy as np
import pandas as pd
import datetime

import warnings
import functools
import inspect

def path_to_text(path,fName,colHeads=None):
    if colHeads is None:
        header = ""
    else:
        header = ",".join(colHeads)
        
    np.savetxt(fName,path,delimiter=",",header=header,fmt="%.6e")
    
    return None

def path_from_text(fName,returnHeads=False):
    df = pd.read_csv(fName,sep=",",index_col=None,header=None)
    
    firstRow = np.array(df.loc[0])
    try:
        firstRow = firstRow.astype(float)
        arr = np.array(df)
        heads = None
    except ValueError:
        arr = np.array(df.loc[1:]).astype(float)
        heads = df.loc[0]
        
    if returnHeads:
        return arr, heads
    else:
        return arr

class ForceLogger:
    #TODO: log interpolators better/at all. Want to allow a link to the dataset(s)
    #interpolated, in case we just use the default data; otherwise, dump the data
    #to the file. Better yet - just make the user link it, in a separate method
    def __init__(self,classInst,logLevel,loggerSettings,fileExt):
        self.loggerSettings = loggerSettings
        defaultSettings = {"writeFreq":50,"logName":None,"writeInterpData":False}
        for s in defaultSettings:
            if s not in self.loggerSettings:
                self.loggerSettings[s] = defaultSettings[s]
        
        self.logLevel = logLevel
        if self.logLevel not in [0,1]:
            raise ValueError("ForceLogger logLevel "+str(self.logLevel)+\
                             " not allowed.")
                
        self.initTime = datetime.datetime.now().isoformat()
        self.classInst = classInst
        os.makedirs("logs",exist_ok=True)
        
        self.loggedVariables = \
            {0:[],1:["points","tangents","springForce","netForce"]}
        varShapes = \
            {0:[],1:4*[(self.loggerSettings["writeFreq"],self.classInst.nPts,self.classInst.nDims)]}
        
        if self.logLevel != 0:
            self.iterCounter = 0
            #Setting the variables that are to be logged at this level
            self.logDict = {}
            for (dsetNm,dsetShape) in zip(self.loggedVariables[self.logLevel],\
                                          varShapes[self.logLevel]):
                self.logDict[dsetNm] = np.zeros(dsetShape)
            
            if self.loggerSettings["logName"] is None:
                self.fileName = "logs/"+self.initTime+fileExt
            else:
                self.fileName = "logs/"+self.loggerSettings["logName"]+fileExt
                        
            #Creating attributes and initializing datasets
            h5File = h5py.File(self.fileName,"w")
            
            #For any nonzero logging level, we'll want these attributes. It's just
            #a question of which datasets we want to store
            # if isinstance(self.classInst.potential,NDInterpWithBoundary):
            #     h5File.create_group("potential")
            #     h5File["potential"].attrs.create("potential",self.classInst.potential.__qualname__)
            #     #TODO: write potential settings here
            # else:
            if hasattr(self.classInst.potential,"__qualname__"):
                nm = self.classInst.potential.__qualname__
            elif hasattr(type(self.classInst.potential),"__qualname__"):
                nm = type(self.classInst.potential).__qualname__
            else:
                nm = "unknown"
                warnings.warn("ForceLogger potential has no attribute __qualname__,"+\
                              " will be logged as unknown name")
            h5File.attrs.create("potential",nm)
            
            h5File.attrs.create("target_func",self.classInst.target_func.__qualname__)
            h5File.attrs.create("target_func_grad",self.classInst.target_func_grad.__qualname__)
            
            #MinimumEnergyPath does not use the inertia tensor; this accounts for that
            if hasattr(self.classInst,"mass"):
                if self.classInst.mass is None:
                    massNm = "constant"
                else:
                    if hasattr(self.classInst.mass,"__qualname__"):
                        massNm = self.classInst.mass.__qualname__
                    elif hasattr(type(self.classInst.mass),"__qualname__"):
                        massNm = type(self.classInst.mass).__qualname__
                    else:
                        warnings.warn("ForceLogger mass has no attribute __qualname__,"+\
                                      " will be logged as unknown name")
                        massNm = "unknown"
                    #TODO: to actually log the mass function, we need to make
                    #it a class, with a __call__ method
                    # if (isinstance(self.classInst.mass,np.ndarray)) and \
                    #     isinstance(self.classInst.mass)
            else:
                massNm = "constant"
            h5File.attrs.create("mass",massNm)
            
            h5File.attrs.create("endpointSpringForce",np.array(self.classInst.endpointSpringForce))
            h5File.attrs.create("endpointHarmonicForce",np.array(self.classInst.endpointHarmonicForce))
            
            h5File.create_group("nebParams")
            for (key, val) in self.classInst.nebParams.items():
                h5File["nebParams"].attrs.create(key,val)
            
            for (nm, arr) in self.logDict.items():
                h5File.create_dataset(nm,arr.shape,maxshape=(None,)+tuple(arr.shape[1:]))
            
            h5File.close()
            
    def log(self,variablesDict):
        if self.logLevel != 0:
            idx = self.iterCounter % self.loggerSettings["writeFreq"]
            for varNm in self.loggedVariables[self.logLevel]:
                self.logDict[varNm][idx] = variablesDict[varNm]
                
            if idx == (self.loggerSettings["writeFreq"] - 1):
                h5File = h5py.File(self.fileName,"a")
                for nm in self.loggedVariables[self.logLevel]:
                    h5File[nm].resize((self.iterCounter+1,)+tuple(h5File[nm].shape[1:]))
                    h5File[nm][-self.loggerSettings["writeFreq"]:] = self.logDict[nm]
                
                h5File.close()
                
            self.iterCounter += 1
        
        return None
    
    def flush(self):
        """
        Note that this must be called, as the self.log only writes the data in
        chunks. Any data falling outside of those chunks will be stored in 
        self.logDict, but not written.

        Parameters
        ----------
        variables : TYPE
            DESCRIPTION.
        variableNames : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.logLevel != 0:
            idx = self.iterCounter % self.loggerSettings["writeFreq"]
            #Only flushes if necessary
            if idx != (self.loggerSettings["writeFreq"] - 1):
                h5File = h5py.File(self.fileName,"a")
                for nm in self.loggedVariables[self.logLevel]:
                    h5File[nm].resize((self.iterCounter,)+tuple(h5File[nm].shape[1:]))
                    h5File[nm][-idx:] = self.logDict[nm][:idx]
                
                h5File.close()
        return None
    
    def write_fire_params(self,tStep,alpha,stepsSinceReset,fireParams):
        if self.logLevel != 0:
            h5File = h5py.File(self.fileName,"a")
            h5File.create_dataset("tStep",data=tStep)
            h5File.create_dataset("alpha",data=alpha)
            h5File.create_dataset("stepsSinceReset",data=stepsSinceReset)
            
            h5File.create_group("fire_params")
            for (key,val) in fireParams.items():
                h5File["fire_params"].attrs.create(key,val)
            
            h5File.close()
        return None
    
    def write_runtime(self,runTime):
        h5File = h5py.File(self.fileName,"a")
        h5File.attrs.create("runTime",runTime)
        h5File.close()
        return None
    
    def write_early_stop_params(self,earlyStopParams):
        if self.logLevel != 0:
            h5File = h5py.File(self.fileName,"a")
            h5File.create_group("early_stop_params")
            for (key, val) in earlyStopParams.items():
                h5File["early_stop_params"].attrs.create(key,val)
            h5File.close()
            
        return None
        
class LoadForceLogger:
    def __init__(self,file):
        allowedExtensions = [".lap",".mep"]
        fileExt = file[-4:]
        if fileExt not in allowedExtensions:
            raise TypeError("File "+str(file)+" has unrecognized extension "+fileExt)
            
        scalarAttrs = ["potential","target_func","target_func_grad","mass"]
        arrayAttrs = ["endpointSpringForce","endpointHarmonicForce"]
                    
        self.fileName = file
        h5File = h5py.File(self.fileName,"r")
        for attr in h5File.attrs:
            if attr in scalarAttrs:
                setattr(self,attr,h5File.attrs[attr])
            elif attr in arrayAttrs:
                setattr(self,attr,np.array(h5File.attrs[attr]))
            else:
                warnings.warn("Attribute "+attr+" not recognized; will not be loaded")
        
        self.nebParams = {}
        for attr in h5File["nebParams"].attrs:
            self.nebParams[attr] = h5File["nebParams"].attrs[attr]
            
        for dset in h5File.keys():
            setattr(self,dset,np.array(h5File[dset]))
        
        h5File.close()
        
class LoadForceLog(LoadForceLogger):
    def __init__(self,file):
        warnings.warn("Deprecating LoadForceLog in favor of LoadForceLogger")
        LoadForceLogger.__init__(self,file)
        
class VerletLogger:
    def __init__(self,vltInst,logLevel):
        self.logLevel = logLevel
        if self.logLevel not in [0,1]:
            raise ValueError("VerletLogger logLevel "+str(self.logLevel)+\
                             " not allowed.")
                
class NDInterpLogger:
    def __init__(self):
        raise NotImplementedError
        #TODO: add logger when the class is instantiated. Then, e.g. ForceLogger
        #can point to these files

class DijkstraLogger:
    def __init__(self,djkInst,logLevel=1,fName=None):
        self.logLevel = logLevel
        if self.logLevel not in [0,1,2]:
            raise ValueError("DijkstraLogger logLevel "+str(self.logLevel)+\
                             " not allowed.")
        
        self.initTime = datetime.datetime.now().isoformat()
        self.djkInst = djkInst
        os.makedirs("logs",exist_ok=True)
        
        if self.logLevel in [1,2]:
            if fName is None:
                self.fileName = "logs/"+self.initTime+".djk"
            else:
                self.fileName = "logs/"+fName+".djk"
            
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
        elif self.logLevel in [1,2]:
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
    
    def finalize(self,runTime,pathAsText=True):
        #TODO: adjust so that the final path is written to a text file after the
        #HDF5 log is written
        # distsDType = np.dtype({"names":["endpoint","dist","strLabel"],\
        #                        "formats":[(float,(self.classInst.nDims,)),float,\
        #                                   h5py.string_dtype("utf-8")]})
        if pathAsText:
            os.makedirs("paths",exist_ok=True)
        
        if self.logLevel == 1:
            # h5File = h5py.File(self.fName,"a")
            # h5File.create_group("endpoints")
            # distsArr = np.zeros(len(distsDict),dtype=distsDType)
            
            nEndpoints = len(self.classInst.endpointIndices)
            padLen = len(str(nEndpoints))
            
            for (keyIter,key) in enumerate(distsDict.keys()):
                strIter = str(keyIter).zfill(padLen)
                # gpNm = "endpoints/"+strIter
                # h5File.create_group(gpNm)
                # h5File[gpNm].attrs.create("endpoint",key)
                # h5File[gpNm].create_dataset("inds",data=minIndsDict[key])
                # h5File[gpNm].create_dataset("points",data=minPathDict[key])
                
                # distsArr[keyIter]["endpoint"] = key
                # distsArr[keyIter]["dist"] = distsDict[key]
                # distsArr[keyIter]["strLabel"] = strIter
                
                if pathAsText:
                    pathTxtName = "paths/"+self.fNameIn+"_endpoint_"+strIter+".txt"
                    path_to_text(minPathDict[key],pathTxtName)
            
            # h5File.attrs.create("runTime",runTime)
            # h5File.create_dataset("dists",data=distsArr)
            
            # h5File.close()
                    
        return None
            
class LoadDijkstraLogger:
    #Maybe can just be a function
    def __init__(self,file):
        if not file.endswith(".djk"):
            raise TypeError("File "+str(file)+" does not have extension .djk")
        
        scalarAttrs = ["runTime","target_func"]
        tupleAttrs = ["initialInds","initialPoint","minimalEndpt"]
        expectedDSets = ["allPathsIndsDict","allowedEndpoints","endpointIndices",\
                         "inertArr","neighborsVisitDict","pathArrDict","potArr",\
                         "tentativeDistance"]
        dsetsDict = {}
        
        h5File = h5py.File(file,"r")
        
        for attr in h5File.attrs:
            if attr in scalarAttrs:
                setattr(self,attr,h5File.attrs[attr])
            elif attr in tupleAttrs:
                setattr(self,attr,tuple(np.array(h5File.attrs[attr])))
            else:
                warnings.warn("Attribute "+attr+" not recognized; will not be loaded")
                
        for d in expectedDSets:
            if d in h5File:
                dsetsDict[d] = np.array(h5File[d])
            else:
                h5File.close()
                raise ValueError("Dataset "+d+" expected but not found")
        
        self.uniqueCoords = [np.array(h5File["uniqueCoords"][c]) for c in h5File["uniqueCoords"]]
        
        h5File.close()
        
        self._set_attrs(dsetsDict)
        
    def _set_attrs(self,dsetsDict):
        #Tested via Spyder console, but not rigorously
        self.allPathsIndsDict = {}
        for (i,p) in enumerate(dsetsDict["allPathsIndsDict"]):
            self.allPathsIndsDict[tuple(p["finalInd"])] = \
                [tuple(val) for val in p["pathInds"][:p["nPts"]]]
                
        self.allowedEndpoints = dsetsDict["allowedEndpoints"]
        self.endpointIndices = [tuple(val) for val in dsetsDict["endpointIndices"]]
        self.inertArr = np.array(dsetsDict["inertArr"])
        
        self.neighborsVisitDict = {}
        for (i,p) in enumerate(dsetsDict["neighborsVisitDict"]):
            self.neighborsVisitDict[tuple(p["key"])] = tuple(p["val"])
            
        self.pathArrDict = {}
        for (i,p) in enumerate(dsetsDict["pathArrDict"]):
            self.pathArrDict[tuple(p["finalPoint"])] = \
                np.array(p["path"][:p["nPts"]])
                
        self.potArr = dsetsDict["potArr"]
        
        self.tentativeDistance = \
            np.ma.masked_array(dsetsDict["tentativeDistance"]["data"],\
                               mask=dsetsDict["tentativeDistance"]["mask"])
            
        return None
    
class DPMLogger:
    def __init__(self,classInst,logLevel=1,fName=None):
        os.makedirs("logs",exist_ok=True)
        
        if fName is None:
            fName = datetime.datetime.now().isoformat()
        self.fName = "logs/"+fName+".dpm"
        self.fNameIn = fName #Could be cleaner -_-
        if logLevel not in [0,1]:
            raise ValueError("logLevel "+str(logLevel)+" not allowed")
        self.logLevel = logLevel
        
        self.classInst = classInst
        self._initialize_log()
        
    def _initialize_log(self):
        if self.logLevel == 1:
            h5File = h5py.File(self.fName,"w")
            
            #Standard logging of the grid data, allowed endpoints, etc.
            h5File.attrs.create("initialInds",np.array(self.classInst.initialInds))
            h5File.attrs.create("initialPoint",np.array(self.classInst.initialPoint))
            h5File.attrs.create("target_func",self.classInst.target_func.__qualname__)
            
            #TODO: allow for naming the coordinates
            h5File.create_group("uniqueCoords")
            for (cIter, coord) in enumerate(self.classInst.uniqueCoords):
                h5File["uniqueCoords"].create_dataset("coord_"+str(cIter),\
                                                      data=np.array(coord))
            h5File.create_dataset("potArr",data=self.classInst.potArr)
            if self.classInst.trimVals[0] is not None:
                h5File["potArr"].attrs.create("minTrim",data=self.classInst.trimVals[0])
            if self.classInst.trimVals[1] is not None:
                h5File["potArr"].attrs.create("maxTrim",data=self.classInst.trimVals[1])
                
            h5File.create_dataset("inertArr",data=self.classInst.inertArr)
            
            h5File.create_dataset("endpointIndices",data=np.array(self.classInst.endpointIndices))
            h5File.create_dataset("allowedEndpoints",data=self.classInst.allowedEndpoints)
            
            #Initializing datasets
            previousIndsArrInit = -1*np.ones(self.classInst.potArr.shape+(self.classInst.nDims,),\
                                             dtype=int)
            h5File.create_dataset("previousIndsArr",data=previousIndsArrInit)
            distArrInit = np.inf*np.ones(self.classInst.potArr.shape)
            h5File.create_dataset("distArr",data=distArrInit)
            
            h5File.close()
        
        return None
    
    def log(self,previousIndsArr,distArr,updateRange):
        if self.logLevel == 1:
            print("Logging slice ",updateRange)
            h5File = h5py.File(self.fName,"a")
            
            slc = (slice(None),slice(*updateRange))+(self.classInst.nDims-2)*(slice(None,),)
            h5File["previousIndsArr"][slc] = previousIndsArr[slc]
            h5File["distArr"][slc] = distArr[slc]
            
            h5File.close()
        
        return None
    
    def finalize(self,minPathDict,minIndsDict,distsDict,runTime,\
                 pathAsText=True):
        distsDType = np.dtype({"names":["endpoint","dist","strLabel"],\
                               "formats":[(float,(self.classInst.nDims,)),float,\
                                          h5py.string_dtype("utf-8")]})
        if pathAsText:
            os.makedirs("paths/dpm",exist_ok=True)
        
        if self.logLevel == 1:
            h5File = h5py.File(self.fName,"a")
            h5File.create_group("endpoints")
            distsArr = np.zeros(len(distsDict),dtype=distsDType)
            
            nEndpoints = len(self.classInst.endpointIndices)
            padLen = len(str(nEndpoints))
            
            for (keyIter,key) in enumerate(distsDict.keys()):
                strIter = str(keyIter).zfill(padLen)
                gpNm = "endpoints/"+strIter
                h5File.create_group(gpNm)
                h5File[gpNm].attrs.create("endpoint",key)
                h5File[gpNm].create_dataset("inds",data=minIndsDict[key])
                h5File[gpNm].create_dataset("points",data=minPathDict[key])
                
                distsArr[keyIter]["endpoint"] = key
                distsArr[keyIter]["dist"] = distsDict[key]
                distsArr[keyIter]["strLabel"] = strIter
                
                if pathAsText:
                    pathTxtName = "paths/dpm/"+self.fNameIn+"_endpoint_"+strIter+".txt"
                    path_to_text(minPathDict[key],pathTxtName)
            
            h5File.attrs.create("runTime",runTime)
            h5File.create_dataset("dists",data=distsArr)
            
            h5File.close()
                    
        return None
    
class LoadDPMLogger:
    def __init__(self,fName):
        if not fName.endswith(".dpm"):
            raise TypeError("File "+str(fName)+" does not have extension .dpm")
        
        scalarAttrs = ["runTime","target_func"]
        tupleAttrs = ["initialInds","initialPoint"]
        expectedDSets = ["allowedEndpoints","distArr","potArr","previousIndsArr","dists"]
        
        dsetsDict = {}
        
        h5File = h5py.File(fName,"r")
        
        for attr in h5File.attrs:
            if attr in scalarAttrs:
                setattr(self,attr,h5File.attrs[attr])
            elif attr in tupleAttrs:
                setattr(self,attr,tuple(np.array(h5File.attrs[attr])))
            else:
                warnings.warn("Attribute "+attr+" not recognized; will not be loaded")
                
        for d in expectedDSets:
            if d in h5File:
                setattr(self,d,np.array(h5File[d]))
            else:
                h5File.close()
                raise ValueError("Dataset "+d+" expected but not found")
        
        self.uniqueCoords = [np.array(h5File["uniqueCoords"][c]) for c in h5File["uniqueCoords"]]
        
        self.pathIndsDict = {}
        self.pathDict = {}
        
        for gp in h5File["endpoints"]:
            key = tuple(np.array(h5File["endpoints"][gp].attrs["endpoint"]))
            self.pathIndsDict[key] = np.array(h5File["endpoints"][gp]["inds"])
            self.pathDict[key] = np.array(h5File["endpoints"][gp]["points"])
        
        h5File.close()
        