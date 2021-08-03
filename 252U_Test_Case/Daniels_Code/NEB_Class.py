import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters, morphology #For minimum finding
from scipy.signal import argrelextrema
import time
import datetime

from scipy.integrate import solve_bvp
from shapely import geometry #Used in initializing the LNEB method on the gs contour
 
#Use Rbf in my NN code, but there are too many points here - Rbf runs out of system
#memory and the Python console crashes
#TODO: namespace stuff here
from scipy.interpolate import interp2d, Rbf
from scipy import interpolate
import sklearn.gaussian_process as gp

import pandas as pd
import h5py
import os
import sys

class Utilities:
    @staticmethod
    def find_local_minimum(arr):
        """
        Returns the indices corresponding to the local minimum values. Taken 
        directly from https://stackoverflow.com/a/3986876
        
        Parameters
        ----------
        arr : Numpy array
            A D-dimensional array.
    
        Returns
        -------
        minIndsOut : Tuple of numpy arrays
            D arrays of length k, for k minima found. Each tuple is all of the minima
            for a coordinate. So, e.g. ([2,3],[0,12],[7,48]) found 2 minima in
            a 3-dimensional array: one at the indices (2,0,7), and another at
            the indices (3,12,48). Returned in this format so that one can call
            xx[minIndsOut] to get the x coordinates of all minima, where xx is a
            meshgrid of x coordinates (an output of np.meshgrid(*args)).
    
        """
        neighborhood = morphology.generate_binary_structure(len(arr.shape),1)
        local_min = (filters.minimum_filter(arr, footprint=neighborhood,\
                                            mode="nearest")==arr)
        
        background = (arr==0)
        #Not sure this is necessary - it doesn't seem to do much on the test
            #data I defined.
        eroded_background = morphology.binary_erosion(background,\
                                                      structure=neighborhood,\
                                                      border_value=1)
            
        detected_minima = local_min ^ eroded_background
        allMinInds = np.vstack(local_min.nonzero())
        
        minIndsOut = tuple([allMinInds[coordIter,:] for \
                            coordIter in range(allMinInds.shape[0])])
        
        return minIndsOut
    
    @staticmethod
    def find_base_dir():
        #Auto-config for my desktop or the HPCC
        possibleHomeDirs = ["~/Research/ActionMinimization/","~/Documents/ActionMinimization/"]
        
        baseDir = None
        for d in possibleHomeDirs:
            if os.path.isdir(os.path.expanduser(d)):
                baseDir = os.path.expanduser(d)
        if baseDir is None:
            sys.exit("Err: base directory not found")
            
        return baseDir
    
    @staticmethod
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
    
    @staticmethod
    def initial_on_contour(coordMeshTuple,zz,nPts,debug=False):
        """
        Connects a straight line between the metastable state and the contour at
        the same energy. WARNING: only useful for leps_plus_ho
        """
        nCoords = len(coordMeshTuple)
        
        minInds = Utilities.find_local_minimum(zz)
        startInds = tuple([minInds[cIter][np.argmax(zz[minInds])] for\
                           cIter in range(nCoords)])
        metaEneg = zz[startInds]
        
        #Pulled from PES code. Doesn't generalize to higher dimensions, but not
        #an important issue rn.
        fig, ax = plt.subplots()
        ax.contourf(coordMeshTuple[0],coordMeshTuple[1],zz)
        contour = ax.contour(coordMeshTuple[0],coordMeshTuple[1],zz,\
                             levels=[metaEneg]).allsegs[0][0].T#Selects the actual curve
        if not debug:
            plt.close(fig)
            
        startPt = np.array([coordMeshTuple[tupInd][startInds] for tupInd in \
                            range(nCoords)])
        line = geometry.LineString(contour.T)
        
        approxFinalPt = np.array([1.5,1])
        point = geometry.Point(*approxFinalPt)
        finalPt = np.array(line.interpolate(line.project(point)))
        # print(point.distance(line))
            
        initialPoints = np.array([np.linspace(startPt[cInd],finalPt[cInd],num=nPts) for\
                                  cInd in range(nCoords)])
        
        return initialPoints
    
    @staticmethod
    def aux_pot(eneg_func,eGS,tol=10**(-2)):
        def pot_out(*coords):
            return eneg_func(*coords) - eGS + tol
        
        return pot_out
    
    @staticmethod
    def const_mass():
        def dummy_mass(*coords):
            return np.ones(coords[0].shape)
        
        return dummy_mass
    
    @staticmethod
    def interpolated_action(eneg_func,mass_func,discretePath,ndims=2,nPts=200):
        if ndims not in discretePath.shape:
            sys.exit("Err: path shape is "+str(discretePath.shape)+"; expected one dimension to be "+str(ndims))
            
        if discretePath.shape[1] == ndims:
            discretePath = discretePath.T
            
        #s=0 means that the interpolation actually passes through all of the points
        tck, u = interpolate.splprep(discretePath,s=0)
        uDense = np.linspace(0,1,num=nPts)
        
        pathOut = interpolate.splev(uDense,tck)
        
        enegs = eneg_func(*pathOut)
        masses = mass_func(*pathOut)
        
        actOut = 0
        for ptIter in range(1,nPts):
            dist2 = 0
            for coordIter in range(ndims):
                dist2 += (pathOut[coordIter][ptIter] - pathOut[coordIter][ptIter-1])**2
            actOut += (np.sqrt(2*masses[ptIter]*enegs[ptIter])+\
                       np.sqrt(2*masses[ptIter-1]*enegs[ptIter-1]))*np.sqrt(dist2)
        actOut = actOut/2
        
        return pathOut, actOut
    
    @staticmethod
    def standard_pes(xx,yy,zz,clipRange=(-5,30)):
        #TODO: pull some (cleaner) options from ML_Funcs_Class
        #Obviously not general - good luck plotting a 3D PES lol
        fig, ax = plt.subplots()
        #USE THIS COLORMAP FOR PESs - has minima in blue and maxima in red
        cf = ax.contourf(xx,yy,zz.clip(clipRange[0],clipRange[1]),\
                         cmap="Spectral_r",levels=np.linspace(clipRange[0],clipRange[1],25))
        plt.colorbar(cf,ax=ax)
        
        ax.set(xlabel=r"$Q_{20}$ (b)",ylabel=r"$Q_{30}$ (b${}^{3/2}$)")
        return fig, ax
    
class CustomInterp2d(interp2d):
    def __init__(self,*args,**kwargs):
        self.kwargs = kwargs
        super(CustomInterp2d,self).__init__(*args,**kwargs)
        self.potential = self.pot_wrapper()
        
    #So that I can pull out a string representation, and kwargs
    def __str__(self):
        return "scipy.interpolate.interp2d"
    
    #Need a wrapper for use in abstracted function calls
    def pot_wrapper(self):
        def potential(*coords):
            #Is a 2d method anyways
            flattenedCoords = [coords[0].flatten(),coords[1].flatten()]
            nCoords = len(coords)
            
            enegOut = np.zeros(flattenedCoords[0].shape)
            for ptIter in range(len(enegOut)):
                enegOut[ptIter] = self.__call__(flattenedCoords[0][ptIter],flattenedCoords[1][ptIter])
            
            return enegOut.reshape(coords[0].shape)
        
        return potential
    
class LocalGPRegressor(gp.GaussianProcessRegressor):
    #WARNING: when written this way, gridPot is actually a keyword-only argument...
    #and yet, it's required.
    def __init__(self,*uniqueGridPts,gridPot,predKWargs={},gpKwargs={}):
        if gridPot is None:
            sys.exit("Err: gridPot must be specified with kwarg")
        super(LocalGPRegressor,self).__init__(**gpKwargs)
        self.uniqueGridPts = uniqueGridPts
        
        self.inDat = np.meshgrid(*uniqueGridPts)
        self.outDat = gridPot
        self.predKWargs = predKWargs
        self._initialize_predkwargs()
        
        self.potential = self.pot_wrapper()
        
    def _initialize_predkwargs(self):
        gridDistance = np.zeros(len(self.uniqueGridPts))
        for (gIter,grid) in enumerate(self.uniqueGridPts):
            #Assumes evenly-spaced grid. Is not quite right, but is close enough
            gridDistance[gIter] = (grid[-1] - grid[0])/len(grid)
        
        defaultKWargs = {"nNeighbors":10,"neighborDist":gridDistance}
        for arg in defaultKWargs.keys():
            if arg not in self.predKWargs.keys():
                self.predKWargs[arg] = defaultKWargs[arg]
                
        return None
    
    def pot_wrapper(self):
        def potential(*coords):
            nCoords = len(self.inDat)
            
            neighborDist = self.predKWargs["neighborDist"]
            nNeighbors = self.predKWargs["nNeighbors"]
            
            if not isinstance(nNeighbors,np.ndarray):
                nNeighbors = nNeighbors*np.ones(nCoords)
            allowedDistance = nNeighbors * neighborDist
            
            flattenedCoords = [c.flatten() for c in coords]
            nPts = len(flattenedCoords[0])
            predDat = np.zeros(nPts)
            
            #TODO: can optimize this a bit by first checking if predPts are close to
            #each other, and using the same fitted GP for them if so
            for ptIter in range(nPts):
                boolList = []
                for dimIter in range(nCoords):
                    locMask = (np.abs(self.inDat[dimIter] - flattenedCoords[dimIter][ptIter]) < \
                               allowedDistance[dimIter])
                    boolList.append(locMask)
                boolMask = np.logical_and.reduce(np.array(boolList))
                usedInds = tuple(np.argwhere(boolMask==True).T)
                
                locInDat = np.array([self.inDat[cIter][usedInds] for\
                                      cIter in range(nCoords)]).T
                locOutDat = self.outDat[usedInds]
                
                locFitGP = self.fit(locInDat,locOutDat)
                
                predDat[ptIter] = \
                    self.predict(np.array([flattenedCoords[cIter][ptIter] for\
                                           cIter in range(nCoords)]).reshape((-1,nCoords)))
            
            return predDat.reshape(coords[0].shape)
        return potential
    
class LepsPot():
    def __init__(self,initialGrid=np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05)),\
                 potParams={},eGS=None,loggingLevel=None,logFile=None):
        # if (loggingLevel is not None) and (logFile is None):
        #     logFile = datetime.datetime.now().isoformat()+"_log.h5"
        
        self.params = potParams
        self._initialize_params()
        
        self.initialGrid = initialGrid
        self.zz = self.leps_plus_ho(*self.initialGrid)
        minInds = Utilities.find_local_minimum(self.zz)
        #Note: only good for this case, where the potential has exactly two minima.
        #This makes it so that the metastable state is at E = 0.
        if eGS is None:
            eGS = np.max(self.zz[minInds])
        
        self.potential = Utilities.aux_pot(self.leps_plus_ho,eGS)
        
    def _initialize_params(self):
        defaultDict = {"a":0.05,"b":0.8,"c":0.05,"dab":4.746,"dbc":4.746,\
                       "dac":3.445,"r0":0.742,"alpha":1.942,"rac":3.742,"kc":0.2025,\
                       "c_ho":1.154}
        for param in defaultDict.keys():
            if param not in self.params.keys():
                self.params[param] = defaultDict[param]
            
        return None
    
    def _q(self,r,d,alpha,r0):
        return d/2*(3/2*np.exp(-2*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))
    
    def _j(self,r,d,alpha,r0):
        return d/4*(np.exp(-2*alpha*(r-r0)) - 6*np.exp(-alpha*(r-r0)))
    
    def leps_pot(self,rab,rbc):
        q = self._q
        j = self._j
                
        rac = rab + rbc
        
        vOut = q(rab,self.params["dab"],self.params["alpha"],self.params["r0"])/(1+self.params["a"]) +\
            q(rbc,self.params["dbc"],self.params["alpha"],self.params["r0"])/(1+self.params["b"]) +\
            q(rac,self.params["dac"],self.params["alpha"],self.params["r0"])/(1+self.params["c"])
        
        jab = j(rab,self.params["dab"],self.params["alpha"],self.params["r0"])
        jbc = j(rbc,self.params["dbc"],self.params["alpha"],self.params["r0"])
        jac = j(rac,self.params["dac"],self.params["alpha"],self.params["r0"])
        
        jTerm = jab**2/(1+self.params["a"])**2+jbc**2/(1+self.params["b"])**2+\
            jac**2/(1+self.params["c"])**2
        jTerm = jTerm - jab*jbc/((1+self.params["a"])*(1+self.params["b"])) -\
            jbc*jac/((1+self.params["b"])*(1+self.params["c"])) -\
            jab*jac/((1+self.params["a"])*(1+self.params["c"]))
        
        vOut = vOut - np.sqrt(jTerm)
        
        return vOut
    
    def leps_plus_ho(self,rab,x):
        """
        LEPs potential plus harmonic oscillator. TODO: add source
        
        Call this function with a numpy array of rab and x:
        
            xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
            zz = leps_plus_ho(xx,yy),
        
        and plot it as
        
            fig, ax = plt.subplots()
            ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")
    
        """
        vOut = self.leps_pot(rab,self.params["rac"]-rab)
        vOut += 2*self.params["kc"]*(rab-(self.params["rac"]/2-x/self.params["c_ho"]))**2
        
        return vOut
    
class SylvesterPot:
    def __init__(self,initialGrid=np.meshgrid(np.arange(-2,2,0.05),np.arange(-3,3,0.05)),\
                 eGS=None):
        self.initialGrid = initialGrid
        self.zz = self.sylvester_pes(*self.initialGrid)
        minInds = Utilities.find_local_minimum(self.zz)
        self.minInds = minInds
        
        #Probably useful in general to select the max, so that there's a contour
        #of eGS around the global minimum
        if eGS is None:
            eGS = np.max(self.zz[minInds])
        
        self.potential = Utilities.aux_pot(self.sylvester_pes,eGS)
        
    def sylvester_pes(self,x,y):
        A = -0.8447
        B = -0.2236
        C = 0.1247
        D = -4.468
        E = 0.02194
        F = 0.3041
        G = 0.1687
        H = 0.4388
        I = -4.713 * 10**(-7)
        J = -1.148 * 10**(-5)
        K = 1.687
        L = -3.062 * 10**(-18)
        M = -9.426 * 10**(-6)
        N = -2.851 * 10**(-16)
        O = 2.313 * 10**(-5)
        
        vOut = A + B*x + C*y + D*x**2 + E*x*y + F*y**2 + G*x**3 + H*x**2*y
        vOut += I*x*y**2 + J*y**3 + K*x**4 + L*x**3*y + M*x**2*y**2 + N*x*y**3 + O*y**4
        
        return vOut
    
class LoggingUtilities():
    #IGNORE THIS - possibly useful abstraction for the future, but not helpful here
    
    #Idea for logging above: perhaps just pull the time as the dataset, then
    #have an attribute for the function call. If the dataset exists, wait
    #the millisecond or so it takes to get the next datetime.datetime.whatever,
    #and write it again. As this is all sequential, won't be an issue (for a while).
    #Assume for now that anything that'll be parallelized will come with its own logging
    #method - perhaps a thread ID
    
    # ###Boilerplate logging
    # allowedLoggingLevels = [None,"debug"]
    # self.logFolder = LoggingUtilities.make_log_directory()
    
    # self.loggingLevel = loggingLevel
    # if self.loggingLevel not in allowedLoggingLevels:
    #     sys.exit("Err: requested logging level "+str(self.loggingLevel)+\
    #              "; allowed levels are "+str(allowedLoggingLevels))
    
    # if self.loggingLevel is None:
    #     self.logFile = None
    # else:
    #     if logFile is None:
    #         self.logFile = \
    #             self.logFolder+datetime.datetime.now().isoformat()+".h5"
    #     else:
    #         if not logFile.startswith("Logs/"):
    #             logFile = "Logs/"+logFile
    #         self.logFile = logFile
    
    # self.loggingBelow = {}
    
    # subClasses = [lnebObj]
    # if self.loggingLevel is not None:
    #     LoggingUtilities.subclass_logging(subClasses,self.loggingBelow,self.logFile)
    
    # ###Ends here
    
    @staticmethod
    def make_log_directory():
        #I thought this'd be more complicated, and hence worthy of abstraction. I guess not lol
        logFolder = "Logs/"
        os.makedirs(logFolder,exist_ok=True)
        
        return logFolder
    
    @staticmethod
    def gp_name(h5File):
        existFlag = True
        
        while existFlag:
            gpNm = datetime.datetime.now().isoformat()
            if gpNm not in h5File.keys():
                existFlag = False
            else:
                time.sleep(0.001)
        h5File.create_group(gpNm)
        
        return gpNm
    
    @staticmethod
    def subclass_logging(subClasses,loggingBelow,logFile):
        for classInstance in subClasses:
            #If object may be logging, check if it definitively is
            if hasattr(classInstance,"loggingLevel"):
                #If object won't be logging, set to false, and dump class again
                if classInstance.loggingLevel is None:
                    #I think all classes have a .__str__ attr, but sometimes it
                    #isn't useful
                    loggingBelow[classInstance.__str__()] = (False,classInstance)
                #If object will be logging, rely on logging on lower level
                else:
                    loggingBelow[classInstance.__str__()] = (True,classInstance)
            #If object doesn't know how to log, dump every attribute of class instance
            else:
                loggingBelow[classInstance.__str__()] = (False,classInstance)
            
            for key in loggingBelow.keys():
                if not loggingBelow[key][0]:
                    if not hasattr(loggingBelow[key][1],"dump_to_file"):
                        print("Warning: method "+str(key)+" has no way to dump to file.")
                    else:
                        loggingBelow[key][1].dump_to_file(logFile)
        return None
    
    @staticmethod
    def logging(loggingLevel,logFile,datDict,strRep):
        if loggingLevel is not None:
            h5File = h5py.File(logFile,mode="a")
            
            gpNm = LoggingUtilities.gp_name(h5File)
            
            for key in datDict.keys():
                h5File[gpNm].create_dataset(key,data=datDict[key])
            
            h5File[gpNm].attrs.create("Function",strRep)
            
            h5File.close()
            
        return None
    
class FileIO(): #Tried using inheritance here, but I think the static methods
                #messed things up for some reason.
    
    @staticmethod
    def read_from_h5(fName,fDir,baseDir=None):
        if baseDir is None:
            baseDir = Utilities.find_base_dir()
        
        datDictOut = {}
        attrDictOut = {}
        
        h5File = h5py.File(baseDir+fDir+fName,"r")
        allDataSets = [key for key in Utilities.h5_get_keys(h5File) if isinstance(h5File[key],h5py.Dataset)]
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
    
    @staticmethod
    def dump_to_hdf5(fname,otpOutputsDict,paramsDictOfDicts):
        h5File = h5py.File(fname+".h5","w")
        for key in otpOutputsDict.keys():
            h5File.create_dataset(key,data=otpOutputsDict[key])
        
        for outerKey in paramsDictOfDicts:
            h5File.create_group(outerKey)
            for key in paramsDictOfDicts[outerKey].keys():
                #Don't know if individual strings are converted automatically
                if isinstance(paramsDictOfDicts[outerKey][key],str):
                    custType = h5py.string_dtype("utf-8")
                else:
                    custType = type(paramsDictOfDicts[outerKey][key])
                h5File[outerKey].attrs.create(key,paramsDictOfDicts[outerKey][key],\
                                                   dtype=custType)
        
        h5File.close()
        
        return None
    
    @staticmethod
    def read_path(fname):
        df = pd.read_csv(fname,sep=",",header=None,index_col=None)
        return np.array(df)
    
    @staticmethod
    def write_path(fname,path,ndims=2):
        #TODO: error handling when path exists
        if ndims not in path.shape:
            sys.exit("Err: path shape is "+str(path.shape)+"; expected one dimension to be "+str(ndims))
            
        if path.shape[0] == ndims:
            path = path.T
        
        df = pd.DataFrame(data=path)
        df.to_csv(fname,sep=",",header=False,index=False)
        
        return None
    
class CustomLogging():
    def __init__(self,loggingLevel):
        allowedLogLevels = [None,"output"]#"output" doesn't keep track of intermediate steps
        assert loggingLevel in allowedLogLevels
        self.loggingLevel = loggingLevel
        if self.loggingLevel == "output":
            self.logDict = {}
            self.outputNms = {}
    
    def update_log(self,strRep,outputTuple,outputNmsTuple,isTuple=True):
        #If returning a single value, set isTuple -> False
        if self.loggingLevel is None:
            return None
        
        if self.loggingLevel == "output":
            if not isTuple:
                outputTuple = (outputTuple,)
                outputNmsTuple = (outputNmsTuple,)
            
            if strRep not in self.logDict:
                self.logDict[strRep] = []
                for t in outputTuple:
                    if isinstance(t,np.ndarray):
                        self.logDict[strRep].append(np.expand_dims(t,axis=0))
                    else:
                        self.logDict[strRep].append([t])
                self.outputNms[strRep] = outputNmsTuple
            else:
                assert len(outputTuple) == len(self.logDict[strRep])
                for (tIter,t) in enumerate(outputTuple):
                    if isinstance(t,np.ndarray):
                        self.logDict[strRep][tIter] = \
                            np.concatenate((self.logDict[strRep][tIter],np.expand_dims(t,axis=0)))
                    else:
                        self.logDict[strRep][tIter].append(t)
                        
        return None
    
    def write_log(self,fName,overwrite=False):
        #WARNING: probably doesn't handle anything that isn't a numpy array, although
            #that's almost all that I intend to log at the moment
        if not hasattr(self,"logDict"):
            return None
        
        if not fName.startswith("Logs/"):
            fName = "Logs/"+fName
        os.makedirs("Logs",exist_ok=True)
        
        if (overwrite) and (os.path.isfile(fName)):
            os.remove(fName)
        
        h5File = h5py.File(fName,"a")
        for key in self.logDict.keys():
            splitKey = key.split(".")
            for (sIter,s) in enumerate(splitKey):
                subGp = "/".join(splitKey[:sIter+1])
                if not subGp in h5File:
                    h5File.create_group(subGp)
            for (oIter,outputNm) in enumerate(self.outputNms[key]):
                h5File[key.replace(".","/")].create_dataset(outputNm,data=self.logDict[key][oIter])
        
        h5File.close()
        
        return None
    
class LineIntegralNeb(CustomLogging):
    def __init__(self,potential,mass,initialPoints,k,kappa,constraintEneg,\
                 targetFunc="trapezoidal_action",loggingLevel=None):
        super().__init__(loggingLevel)#Believe it or not - still unclear what this does...
        
        targetFuncDict = {"trapezoidal_action":self._construct_trapezoidal_action}
        if targetFunc not in targetFuncDict.keys():
            sys.exit("Err: requested target function "+str(targetFunc)+\
                     "; allowed functions are "+str(targetFuncDict.keys()))
                
        self.k = k
        self.kappa = kappa
        self.constraintEneg = constraintEneg
        
        self.nCoords, self.nPts = initialPoints.shape
        self.potential = potential
        self.mass = mass
        self.initialPoints = initialPoints
        
        #Is an approximation of the integral
        self.target_func = targetFuncDict[targetFunc]()
        self.target_func(*initialPoints)
        
    def __str__(self):
        return "LineIntegralNeb"
    
    def _construct_trapezoidal_action(self):
        def trapezoidal_action(*path,enegs=None,masses=None):
            #In case I feed in a tuple, as seems likely given how I handle passing
            #coordinates around
            if not isinstance(path,np.ndarray):
                path = np.array(path)
                
            if enegs is None:
                enegs = self.potential(*path)
            else:
                assert enegs.shape[0] == path.shape[1]
            #Will add error checking here later - still not general with a nonconstant mass
            if masses is None:
                masses = self.mass(*path)
            sqrtTerm = np.sqrt(2*masses*enegs)
            
            retVal = 0
            for ptIter in np.arange(1,self.nPts):
                distVal = np.linalg.norm(path[:,ptIter]-path[:,ptIter-1])
                retVal += (sqrtTerm[ptIter]+sqrtTerm[ptIter-1])*distVal
                
            retVal = retVal/2
                
            return retVal, enegs, masses
        return trapezoidal_action
    
    def _compute_tangents(self,currentPts,energies):
        strRep = "LineIntegralNeb._compute_tangents"
        
        tangents = np.zeros((self.nCoords,self.nPts))
        for ptIter in range(1,self.nPts-1): #Range selected to exclude endpoints
            tp = currentPts[:,ptIter+1] - currentPts[:,ptIter]
            tm = currentPts[:,ptIter] - currentPts[:,ptIter-1]
            dVMax = np.max(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
            dVMin = np.min(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
                
            if (energies[ptIter+1] > energies[ptIter]) and \
                (energies[ptIter] > energies[ptIter-1]):
                tangents[:,ptIter] = tp
            elif (energies[ptIter+1] < energies[ptIter]) and \
                (energies[ptIter] < energies[ptIter-1]):
                tangents[:,ptIter] = tm
            elif energies[ptIter+1] > energies[ptIter-1]:
                tangents[:,ptIter] = tp*dVMax + tm*dVMin
            #Paper gives this as just <, not <=. Probably won't come up...
            # elif energies[ptIter+1] <= energies[ptIter-1]:
            else:
                tangents[:,ptIter] = tp*dVMin + tm*dVMax
                
            #Normalizing vectors
            tangents[:,ptIter] = tangents[:,ptIter]/np.sqrt(np.dot(tangents[:,ptIter],tangents[:,ptIter]))
        
        ret = tangents
        outputNms = "tangents"
        self.update_log(strRep,ret,outputNms,isTuple=False)
        
        return ret
    
    def _negative_gradient(self,points,enegsOnPath,massesOnPath):
        strRep = "LineIntegralNeb._negative_gradient"
        
        eps = 10**(-8)
        
        trueForce = np.zeros(points.shape)
        negIntegGrad = np.zeros(points.shape)
        
        actionOnPath, _, _ = self.target_func(*points,enegs=enegsOnPath,masses=massesOnPath)
        
        for ptIter in range(self.nPts):
            for coordIter in range(self.nCoords):
                steps = points.copy()
                steps[coordIter,ptIter] += eps
                
                enegsAtStep = enegsOnPath.copy()
                enegsAtStep[ptIter] = self.potential(*steps[:,ptIter])
                
                massesAtStep = massesOnPath.copy()
                massesAtStep[ptIter] = self.mass(*steps[:,ptIter])
                
                actionAtStep, _, _ = self.target_func(*steps,enegs=enegsAtStep,\
                                                      masses=massesAtStep)
                trueForce[coordIter,ptIter] = (enegsAtStep[ptIter] - enegsOnPath[ptIter])/eps
                negIntegGrad[coordIter,ptIter] = (actionAtStep - actionOnPath)/eps
        
        #Want the negative gradient of both terms
        trueForce = -trueForce
        negIntegGrad = -negIntegGrad
        
        ret = (trueForce, negIntegGrad)
        outputNms = ("trueForce", "negIntegGrad")
        self.update_log(strRep,ret,outputNms)
        
        return ret
    
    def _spring_force(self,points,tangents):
        strRep = "LineIntegralNeb._spring_force"
        
        diffArr = np.array([points[:,i+1] - points[:,i] for i in range(points.shape[1]-1)]).T
        diffScal = np.array([np.linalg.norm(diffArr[:,i]) for i in range(diffArr.shape[1])])
        
        springForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            springForce[:,i] = self.k*(diffScal[i] - diffScal[i-1])*tangents[:,i]
            
        #Zeroed out here to match Eric's code
        springForce[:,0] = np.zeros(2)#self.k*diffArr[:,0]
        springForce[:,-1] = np.zeros(2)#-self.k*diffArr[:,-1] #Added minus sign here to fix endpoint behavior
        
        ret = springForce
        outputNms = "springForce"
        self.update_log(strRep,ret,outputNms,isTuple=False)
        
        return ret
    
    def compute_force(self,points):
        strRep = "LineIntegralNeb.compute_force"
        
        integVal, energies, masses = self.target_func(*points)
        
        tangents = self._compute_tangents(points,energies)
        trueForce, negIntegGrad = self._negative_gradient(points,energies,masses)
        
        #Note: don't care about the tangents on the endpoints; they don't show up
        #in the net force
        perpForce = negIntegGrad - tangents*(np.array([np.dot(negIntegGrad[:,i],tangents[:,i]) \
                                                       for i in range(points.shape[1])]))
        springForce = self._spring_force(points,tangents)
        
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            netForce[:,i] = perpForce[:,i] + springForce[:,i]
            
        normForce = trueForce[:,0]/np.linalg.norm(trueForce[:,0])
        netForce[:,0] = springForce[:,0] - \
            (np.dot(springForce[:,0],normForce)-\
              self.kappa*(energies[0]-self.constraintEneg))*normForce
            
        normForce = trueForce[:,-1]/np.linalg.norm(trueForce[:,-1])
        netForce[:,-1] = springForce[:,-1] - \
            (np.dot(springForce[:,-1],normForce)-\
              self.kappa*(energies[-1]-self.constraintEneg))*normForce
        
        ret = netForce
        outputNms = "netForce"
        self.update_log(strRep,ret,outputNms,isTuple=False)
        
        return ret
    
class MinimizationAlgorithms(CustomLogging):
    def __init__(self,lnebObj,actionFunc="trapezoid",loggingLevel=None):
        super().__init__(loggingLevel)
        
        actionFuncsDict = {"trapezoid":lnebObj._construct_trapezoidal_action}
        
        self.lneb = lnebObj
        self.action_func = actionFuncsDict[actionFunc]()
    
    def verlet_minimization(self,maxIters=1000,tStep=0.05):
        strRep = "MinimizationAlgorithms.verlet_minimization"
        
        allPts = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        
        allPts[0,:,:] = self.lneb.initialPoints
        
        for step in range(0,maxIters):
            force = self.lneb.compute_force(allPts[step,:,:])
            allForces[step,:,:] = force
            allPts[step+1,:,:] = allPts[step,:,:] + 1/2*force*tStep**2
            
        actions = np.array([self.action_func(pts) for pts in allPts[:]])
        
        ret = (allPts, allForces, actions)
        outputNms = ("allPts","allForces","actions")
        self.update_log(strRep,ret,outputNms)
        
        return ret
    
    def verlet_minimization_v2(self,maxIters=1000,tStep=0.05):
        strRep = "MinimizationAlgorithms.verlet_minimization_v2"
        
        allPts = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        allVelocities = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        
        allPts[0,:,:] = self.lneb.initialPoints
        allForces[0,:,:] = self.lneb.compute_force(allPts[0,:,:])
        
        for step in range(0,maxIters):
            #Velocity update taken from "Classical and Quantum Dynamics in Condensed Phase Simulations",
            #page 397
            for ptIter in range(self.lneb.nPts):
                product = np.dot(allVelocities[step,:,ptIter],allForces[step,:,ptIter])
                if product > 0:
                    vProj = \
                        product*allForces[step,:,ptIter]/np.dot(allForces[step,:,ptIter],allForces[step,:,ptIter])
                else:
                    vProj = np.zeros(self.lneb.nCoords)
                allVelocities[step+1,:,ptIter] = vProj + allForces[step,:,ptIter]*tStep
            allPts[step+1] = allPts[step] + allVelocities[step+1]*tStep+1/2*allForces[step]*tStep**2
            allForces[step+1] = self.lneb.compute_force(allPts[step+1])
                
        actions = np.array([self.action_func(*pts)[0] for pts in allPts[:]])
        
        ret = (allPts, allVelocities, allForces, actions)
        outputNms = ("allPts","allVelocities","allForces","actions")
        self.update_log(strRep,ret,outputNms)
        
        return allPts, allVelocities, allForces, actions
    
    def verlet_minimization_v3(self,maxIters=1000,tStep=0.05):
        strRep = "MinimizationAlgorithms.verlet_minimization_v3"
        
        allPts = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        allVelocities = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        
        allPts[0,:,:] = self.lneb.initialPoints
        allForces[0,:,:] = self.lneb.compute_force(allPts[0,:,:])
        
        for step in range(0,maxIters):
            allPts[step+1] = allPts[step] + allVelocities[step]*tStep+1/2*allForces[step]*tStep**2
            allForces[step+1] = self.lneb.compute_force(allPts[step+1])
            allVelocities[step+1] = allVelocities[step]+(allForces[step]+allForces[step+1])/2*tStep
            
            for ptIter in range(self.lneb.nPts):
                if np.dot(allVelocities[step+1,:,ptIter],allForces[step+1,:,ptIter])<=0:
                    allVelocities[step+1,:,ptIter] = np.zeros(self.lneb.nCoords)
                    # allVelocities[step+1,:,ptIter] = \
                    #     np.dot(allVelocities[step+1,:,ptIter],allForces[step+1,:,ptIter])*\
                    #         allForces[step+1,:,ptIter]/np.dot(allForces[step+1,:,ptIter],allForces[step+1,:,ptIter])
                # else:

                
        actions = np.array([self.action_func(pts) for pts in allPts[:]])
        
        return allPts, allVelocities, allForces, actions, strRep
    
    def gradient_descent(self,maxIters=1000,tStep=0.05):
        #Very clearly does not work...
        allPts = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        allForces = np.zeros((maxIters+1,self.lneb.nCoords,self.lneb.nPts))
        
        allPts[0] = self.lneb.initialPoints
        
        for step in range(0,maxIters):
            allForces[step] = self.lneb.compute_force(allPts[step])
            allPts[step+1] = allPts[step] + tStep * allForces[step]
            
        allForces[-1] = self.lneb.compute_force(allPts[-1])
        actions = np.array([self.action_func(pts) for pts in allPts[:]])
        
        return allPts, allForces, actions
    
    def _el_rhs(self,indepVar,depVar):
        eps = 10**(-6)
        
        #depVar should be of shape (4,nPts) - 4 for (x0,x1,z0,z1); nPts for the size of the grid.
        #Here, zi = dxi/dt.
        funcOut = np.zeros(depVar.shape)
        #funcOut row order is (x0',x1',z0',z1') = (z0,z1,z0',z1')
        funcOut[0:2,:] = depVar[2:,:]
        
        #Not general here or anywhere else
        targEval = self.lneb.targetFunc(*depVar[0:2])
        targGrad = np.zeros(depVar.shape)
        for coordIter in range(self.lneb.nCoords):
            steps = depVar[0:2].copy()
            steps[coordIter,:] += eps
            evalAtSteps, _, _ = self.lneb.target_func(*steps)
            targGrad[coordIter,:] = (evalAtSteps - targetFuncEval)/eps
        
        r = np.sqrt(depVar[2,:]**2 + depVar[3,:]**2)
        
        for gradIter in range(2):
            funcOut[gradIter+2,:] = 1/targEval*targGrad[gradIter,:]*r**2 - \
                depVar[gradIter+2,:]/targEval*(depVar[2,:]*targGrad[0,:]+depVar[3,:]*targGrad[1,:]) + \
                    depVar[gradIter+2,:]**2/r**2
        
        return funcOut
    
    def euler_lagrange(self):
        
        return None
    
# class EulerLagrangeSolver:
#     def __init__(self,nPts=12,potential=leps_plus_ho,initialPoints=None,gsEneg=None,\
#                  subtractEGS=True):
#         self.nPts = nPts
#         self.coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,2,0.05))
#         self.zz = potential(*self.coordMeshTuple)
        
#         if initialPoints is None:
#             self.initialPoints = self._initial_points()
#         else:
#             self.initialPoints = initialPoints
            
#         if subtractEGS and (gsEneg is None):
#             self.gsEneg = np.min(potential(*self.initialPoints))
#         elif subtractEGS and (gsEneg is not None):
#             self.gsEneg = gsEneg
#         else:
#             self.gsEneg = 0
            
#         self.potential = self._aux_pot(potential,self.gsEneg)
            
#     def _initial_points(self):
#         minInds = find_local_minimum(self.zz)
        
#         initialPoints = np.array([np.linspace(cmesh[minInds][0],cmesh[minInds][1],num=self.nPts) for \
#                                   cmesh in self.coordMeshTuple])
        
#         return initialPoints
    
#     def _aux_pot(self,potential,egs,tol=10**(-4)):
#         #Tol is so that the potential at the ground state isn't exactly 0 (have to
#         #divide by V in the EL equations)
#         def pot(x,y):
#             return potential(x,y) - egs + tol
        
#         return pot
        
#     def _finite_difference(self,coords):
#         eps = 10**(-6)
        
#         potAtPt = self.potential(coords[0,:],coords[1,:])
#         potAtStep = np.zeros((2,coords.shape[1]))
#         for i in range(2):
#             for coordIter in range(coords.shape[1]):
#                 locCoord = coords.copy()
#                 locCoord[i,coordIter] += eps
#                 potAtStep[i,coordIter] = self.potential(locCoord[0,coordIter],locCoord[1,coordIter])
        
#         grad = (potAtStep - potAtPt)/eps
        
#         return grad
        
#     def rhs(self,indepVar,depVar):
#         #depVar should be of shape (4,nPts) - 4 for (x0,x1,z0,z1); nPts for the size of the grid.
#         #Here, zi = dxi/dt.
#         funcOut = np.zeros(depVar.shape)
#         #funcOut row order is (x0',x1',z0',z1') = (z0,z1,z0',z1')
#         funcOut[0:2,:] = depVar[2:,:]
        
#         #Not general here or anywhere else
#         potEval = self.potential(depVar[0,:],depVar[1,:])
#         potGrad = self._finite_difference(depVar[0:2,:])
        
#         r = np.sqrt(depVar[2,:]**2 + depVar[3,:]**2)
        
#         for gradIter in range(2):
#             funcOut[gradIter+2,:] = 1/potEval*potGrad[gradIter,:]*r**2 - \
#                 depVar[gradIter+2,:]/potEval*(depVar[2,:]*potGrad[0,:]+depVar[3,:]*potGrad[1,:]) + \
#                     depVar[gradIter+2,:]**2/r**2
        
#         return funcOut
    
#     def bc(self,ya,yb):
#         #ya is right node, yb is left node. both of shape (4,), for (x0,x1,z0,z1)
#         return np.array([ya[0] - self.initialPoints[0,0],ya[1] - self.initialPoints[-1,0],\
#                          yb[0] - self.initialPoints[0,-1],yb[1] - self.initialPoints[-1,-1]])
    
#     def plot_pes(self,curves=[],labels=[],fName=None):
#         fig, ax = plt.subplots()
        
#         ax.contour(*self.coordMeshTuple,self.zz,np.arange(-10,70,1),colors="k")
        
#         if curves: #Checks if list is nonempty
#             for (cIter,curve) in enumerate(curves):
#                 if labels:
#                     lab = labels[cIter]
#                 else:
#                     lab = None
#                 ax.plot(curve[0,:],curve[1,:],ls="-",marker="o",label=lab)
                
#         ax.set_xlabel(r"$r_{AB}$")
#         ax.set_ylabel(r"$x$")
        
#         if labels:
#             ax.legend()
                
#         if fName is not None:
#             fig.savefig(fName+".pdf")
        
#         return None    
    
class Utilities_Validation:
    @staticmethod
    def val_find_local_minimum():
        dsets, attrs = \
            FileIO.read_from_h5("Test_PES.h5","252U_Test_Case/Daniels_Code/Test_Files/")
                
        minInds = Utilities.find_local_minimum(dsets["PES"])
        fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],dsets["PES"])
        
        ax.scatter(dsets["Q20"][minInds],dsets["Q30"][minInds],color="k",marker="x")
        ax.set(xlim=(dsets["Q20"].min(),dsets["Q20"].max()),\
               ylim=(dsets["Q30"].min(),dsets["Q30"].max()))
            
        testFolder = "Test_Outputs/Utilities/"
        if not os.path.isdir(testFolder):
            os.makedirs(testFolder)
            
        fig.savefig(testFolder+"val_find_local_minimum.pdf")
        
        return None
    
    @staticmethod
    def val_interpolated_action():
        testFile = "Test_Files/Test_Path.txt"
        path = FileIO.read_path(testFile)
        
        spot = SylvesterPot()
        smoothedValues, action = \
            Utilities.interpolated_action(spot.potential,Utilities.const_mass(),path)
        
        fig, ax = Utilities.standard_pes(spot.initialGrid[0],spot.initialGrid[1],
                                         spot.potential(*spot.initialGrid),\
                                         clipRange=(-1,10))
        
        ax.plot(path[:,0],path[:,1],ls="-",marker="x",color="k")
        ax.plot(*smoothedValues,ls="-",color="red")
        
        #Convergence of action w/ increasing number of points
        nPtsRange = np.linspace(25,500,num=30,dtype=int)
        actionVals = np.zeros(nPtsRange.shape)
        
        _, baseAction = \
            Utilities.interpolated_action(spot.potential,Utilities.const_mass(),\
                                              path,nPts=path.shape[0])
        
        for (nIter,nPts) in enumerate(nPtsRange):
            _, actionVals[nIter] = \
                Utilities.interpolated_action(spot.potential,Utilities.const_mass(),\
                                              path,nPts=nPts)
            
        fig, ax = plt.subplots()
        ax.plot(nPtsRange,actionVals,marker=".")
        ax.axhline(baseAction,color="red")
        ax.text(nPtsRange[-1]-150,baseAction+0.0005,"Discrete (22 Points)")
        ax.axvline(200,color="black")
        ax.text(205,baseAction+0.004,"Default Number of Points",rotation=-90)
        ax.set_xlim(nPtsRange[0],nPtsRange[-1])
        
        ax.set(xlabel="Number of Interpolation Points",ylabel="Trapezoidal Action",\
               title="Discretized Action Integral\nVersus Number of Points")
        
        testFolder = "Test_Outputs/Utilities/"
        if not os.path.isdir(testFolder):
            os.makedirs(testFolder)
            
        fig.savefig(testFolder+"val_interpolated_action.pdf")
            
        return None
    
class LineIntegralNeb_Validation:
    @staticmethod
    def _q(r,d,alpha,r0):
        return d/2*(3/2*np.exp(-2*alpha*(r-r0)) - np.exp(-alpha*(r-r0)))
    
    @staticmethod
    def _j(r,d,alpha,r0):
        return d/4*(np.exp(-2*alpha*(r-r0)) - 6*np.exp(-alpha*(r-r0)))
    
    @staticmethod
    def leps_pot(rab,rbc):
        q = LineIntegralNeb_Validation._q
        j = LineIntegralNeb_Validation._j
        
        a = 0.05
        b = 0.8
        c = 0.05
        dab = 4.746
        dbc = 4.746
        dac = 3.445
        r0 = 0.742
        alpha = 1.942
        
        rac = rab + rbc
        
        vOut = q(rab,dab,alpha,r0)/(1+a) + q(rbc,dbc,alpha,r0)/(1+b) + q(rac,dac,alpha,r0)/(1+c)
        
        jab = j(rab,dab,alpha,r0)
        jbc = j(rbc,dbc,alpha,r0)
        jac = j(rac,dac,alpha,r0)
        
        jTerm = jab**2/(1+a)**2+jbc**2/(1+b)**2+jac**2/(1+c)**2
        jTerm = jTerm - jab*jbc/((1+a)*(1+b)) - jbc*jac/((1+b)*(1+c)) - jab*jac/((1+a)*(1+c))
        
        vOut = vOut - np.sqrt(jTerm)
        
        return vOut
    
    @staticmethod
    def leps_plus_ho(rab,x):
        """
        Call this function with a numpy array of rab and x:
        
            xx, yy = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,2,0.01))
            zz = leps_plus_ho(xx,yy),
        
        and plot it as
        
            fig, ax = plt.subplots()
            ax.contour(xx,yy,zz,np.arange(-10,70,1),colors="k")
    
        """
        rac = 3.742
        vOut = LineIntegralNeb_Validation.leps_pot(rab,rac-rab)
        
        kc = 0.2025
        c = 1.154
        
        vOut += 2*kc*(rab-(rac/2-x/c))**2
        
        return vOut
    
    @staticmethod
    def _construct_initial_points():
        nPts = 12
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = LineIntegralNeb_Validation.leps_plus_ho(*coordMeshTuple)
        
        minInds = Utilities.find_local_minimum(zz)
        
        initialPoints = np.array([np.linspace(cmesh[minInds][0],\
                                              cmesh[minInds][1],num=nPts) for \
                                  cmesh in coordMeshTuple])
        return initialPoints
    
    @staticmethod
    def _get_egs():
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = LineIntegralNeb_Validation.leps_plus_ho(*coordMeshTuple)
        
        return np.min(zz)
    
    @staticmethod
    def _aux_pot(eGS):
        def pot_out(*coords):
            #Tolerance is necessary in case I don't get exactly the ground state - then,
            #the potential evaluates to be negative, and issues occur
            return LineIntegralNeb_Validation.leps_plus_ho(*coords) - eGS + 10**(-2)
        
        return pot_out
    
    @staticmethod
    def _define_mass():
        def dummy_mass(*coords):
            return np.ones(coords[0].shape)
        
        return dummy_mass
    
    @staticmethod
    def val___init__():
        print(75*"-") #Approximate width of default console in Spyder
        print("Test: val___init__")
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 1
        kappa = 1
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        return None
    
    @staticmethod
    def val__negative_gradient():
        print(75*"-")
        print("Test: val__negative_gradient")
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        print("initialPoints")
        print(initialPoints)
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 1
        kappa = 1
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        targetFuncEval, enegs, masses = lneb.target_func(*lneb.initialPoints)
        normedDistVec, negTargGrad, negIntegGrad = \
            lneb._negative_gradient(lneb.initialPoints,targetFuncEval)
        
        print("negTargGrad:")
        print(negTargGrad)
        print("negIntegGrad:")
        print(negIntegGrad)
        print("normedDistVec:")
        print(normedDistVec)
        
        return None
    
    @staticmethod
    def val__spring_force():
        print(75*"-")
        print("Test: val__spring_force")
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 1
        kappa = 1
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        targetFuncEval, enegs, masses = lneb.target_func(*lneb.initialPoints)
        normedDistVec, negTargGrad, negIntegGrad = \
            lneb._negative_gradient(lneb.initialPoints,targetFuncEval)
        
        k = 1
        
        springForce = lneb._spring_force(initialPoints,-normedDistVec)
        print(springForce)
        
        return None
    
    @staticmethod
    def val__compute_tangents():
        print(75*"-")
        print("Test: val__compute_tangents")
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 1
        kappa = 1
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        targetFuncEval, enegs, masses = lneb.target_func(*lneb.initialPoints)
        
        tangents = lneb._compute_tangents(initialPoints,enegs)
        print(tangents)
        
        return None
    
    @staticmethod
    def val_compute_force():
        print(75*"-")
        print("Test: val_compute_force")
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 1
        kappa = 1
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        targetFuncEval, enegs, masses = lneb.target_func(*lneb.initialPoints)
        normedDistVec, negTargGrad, negIntegGrad = \
            lneb._negative_gradient(lneb.initialPoints,targetFuncEval)
        
        force = lneb.compute_force(initialPoints)
        
        print(force)
        
        return None
    
    @staticmethod
    def val_trapezoidal_action():
        print(75*"-")
        print("Test: val_trapezoidal_action")
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 1
        kappa = 1
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        action = lneb.trapezoidal_action(lneb.initialPoints)
        print(action)
        
        return None
    
class MinimizationAlgorithms_Validation:
    @staticmethod
    def val_verlet_minimization():
        print(75*"-")
        print("Test: val_verlet_minimization")
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 20
        kappa = 20
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        maxIters = 5000
        allPts, allForces, actions = \
            MinimizationAlgorithms(lneb).verlet_minimization(maxIters=maxIters,tStep=0.1)
            
        print("See plots")
        fig, ax = plt.subplots()
        ax.plot(np.arange(actions.shape[0]),actions)
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = lneb.potential(*coordMeshTuple)
        
        fig, ax = plt.subplots()
        cf = ax.contourf(coordMeshTuple[0],coordMeshTuple[1],zz.clip(max=10),np.arange(-10,70,1),\
                         cmap="gist_rainbow",levels=np.arange(0,10.5,0.5))
        plt.colorbar(cf,ax=ax)
        ax.contour(coordMeshTuple[0],coordMeshTuple[1],zz,levels=[constraintEneg],colors="k")
        for i in range(0,maxIters,50):
            ax.plot(allPts[i,0,:],allPts[i,1,:],ls="-",marker="o")
            
        # ax.scatter(initialPoints[0,0],initialPoints[1,0],marker="x",c="k")
        
        return None
    
    @staticmethod
    def val_verlet_minimization_from_contour():
        print(75*"-")
        print("Test: val_verlet_minimization_from_contour")
        
        nPts = 12
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = LineIntegralNeb_Validation.leps_plus_ho(*coordMeshTuple)
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = Utilities.initial_on_contour(coordMeshTuple,zz,nPts)
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 20
        kappa = 20
        constraintEneg = potential(*initialPoints)[0]
        print("Constraining energy to %.3f" %constraintEneg)
        print("Initial points energy:")
        print(potential(*initialPoints))
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        maxIters = 500
        allPts, allForces, actions = \
            MinimizationAlgorithms(lneb).verlet_minimization(maxIters=maxIters,tStep=0.1)
            
        print("See plots")
        fig, ax = plt.subplots()
        ax.plot(np.arange(actions.shape[0]),actions)
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = lneb.potential(*coordMeshTuple)
        
        fig, ax = plt.subplots()
        cf = ax.contourf(coordMeshTuple[0],coordMeshTuple[1],zz.clip(max=10),np.arange(-10,70,1),\
                         cmap="gist_rainbow",levels=np.arange(0,10.5,0.5))
        plt.colorbar(cf,ax=ax)
        
        denseMeshTuple = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,3.8,0.01))
        denseZZ = lneb.potential(*denseMeshTuple)
        ax.contour(denseMeshTuple[0],denseMeshTuple[1],denseZZ,levels=[constraintEneg],colors="k")
        
        for i in range(0,maxIters,int(maxIters/10)):
            ax.plot(allPts[i,0,:],allPts[i,1,:],ls="-",marker="o")
            
        # ax.scatter(initialPoints[0,0],initialPoints[1,0],marker="x",c="k")
        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(allPts[:,0,0],allPts[:,1,0],ls="-",marker="o")
        ax[1].plot(np.arange(maxIters+1),lneb.potential(allPts[:,0,0],allPts[:,1,0]))
        
        return None
    
    @staticmethod
    def val_verlet_minimization_v2():
        print(75*"-")
        print("Test: val_verlet_minimization_v2")
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = LineIntegralNeb_Validation._construct_initial_points()
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 20
        kappa = 20
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        maxIters = 5000
        allPts, allVelocities, allForces, actions = \
            MinimizationAlgorithms(lneb).verlet_minimization_v2(maxIters=maxIters,tStep=0.1)
            
        print("See plots")
        fig, ax = plt.subplots()
        ax.plot(np.arange(actions.shape[0]),actions)
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = lneb.potential(*coordMeshTuple)
        
        fig, ax = plt.subplots()
        cf = ax.contourf(coordMeshTuple[0],coordMeshTuple[1],zz.clip(max=10),np.arange(-10,70,1),\
                         cmap="gist_rainbow",levels=np.arange(0,10.5,0.5))
        plt.colorbar(cf,ax=ax)
        ax.contour(coordMeshTuple[0],coordMeshTuple[1],zz,levels=[constraintEneg],colors="k")
        for i in range(0,maxIters,50):
            ax.plot(allPts[i,0,:],allPts[i,1,:],ls="-",marker="o")
            
        # ax.scatter(initialPoints[0,0],initialPoints[1,0],marker="x",c="k")
        
        return None
    
    @staticmethod
    def val_verlet_minimization_v2_from_contour():
        print(75*"-")
        print("Test: val_verlet_minimization_v2_from_contour")
        
        nPts = 12
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = LineIntegralNeb_Validation.leps_plus_ho(*coordMeshTuple)
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = Utilities.initial_on_contour(coordMeshTuple,zz,nPts)
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 20
        kappa = 20
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        maxIters = 1000
        allPts, allVelocities, allForces, actions = \
            MinimizationAlgorithms(lneb).verlet_minimization_v2(maxIters=maxIters,tStep=0.1)
            
        print("See plots")
        fig, ax = plt.subplots()
        ax.plot(np.arange(actions.shape[0]),actions)
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = lneb.potential(*coordMeshTuple)
        
        fig, ax = plt.subplots()
        cf = ax.contourf(coordMeshTuple[0],coordMeshTuple[1],zz.clip(max=10),np.arange(-10,70,1),\
                         cmap="gist_rainbow",levels=np.arange(0,10.5,0.5))
        plt.colorbar(cf,ax=ax)
        
        denseMeshTuple = np.meshgrid(np.arange(0,4,0.01),np.arange(-2,3.8,0.01))
        denseZZ = lneb.potential(*denseMeshTuple)
        ax.contour(denseMeshTuple[0],denseMeshTuple[1],denseZZ,levels=[constraintEneg],colors="k")
        
        for i in range(0,maxIters,50):
            ax.plot(allPts[i,0,:],allPts[i,1,:],ls="-",marker="o")
            
        return None
    
    @staticmethod
    def val_euler_lagrange():
        print(75*"-")
        print("Test: val_euler_lagrange")
        
        nPts = 12
        
        coordMeshTuple = np.meshgrid(np.arange(0,4,0.05),np.arange(-2,3.8,0.05))
        zz = LineIntegralNeb_Validation.leps_plus_ho(*coordMeshTuple)
        
        eGS = LineIntegralNeb_Validation._get_egs()
        potential = LineIntegralNeb_Validation._aux_pot(eGS)
        
        initialPoints = Utilities.initial_on_contour(coordMeshTuple,zz,nPts)
        
        mass_test = LineIntegralNeb_Validation._define_mass()
        k = 20
        kappa = 20
        constraintEneg = potential(*initialPoints)[0]
        lneb = LineIntegralNeb(potential,mass_test,initialPoints,k,kappa,constraintEneg)
        
        return None

def main():
    startTime = time.time()
    
    lnebParamsDict = {"nPts":22,"k":10,"kappa":10}
    
    fDir = "252U_Test_Case/"
    fName = "252U_PES.h5"
    dsets, attrs = FileIO.read_from_h5(fName,fDir)
    
    #Only getting the unique values
    q20Vals = dsets["Q20"][:,0]
    q30Vals = dsets["Q30"][0]
    
    custInterp2d = CustomInterp2d(q20Vals,q30Vals,dsets["PES"].T,kind="quintic")
    potential = custInterp2d.potential
    
    interpArgsDict = custInterp2d.kwargs
    interpArgsDict["function"] = custInterp2d.__str__()
    
    nPts = lnebParamsDict["nPts"]
    initialPoints = np.array((np.linspace(27,185,nPts),np.linspace(0,16.2,nPts)))
    initialEnegs = potential(*initialPoints)
    constraintEneg = initialEnegs[0]
    
    lnebParamsDict["constraintEneg"] = constraintEneg
    print("Constraining to energy %.3f" % constraintEneg)
    
    k = lnebParamsDict["k"]
    kappa = lnebParamsDict["kappa"]
    lneb = LineIntegralNeb(potential,Utilities.const_mass(),initialPoints,k,kappa,constraintEneg)
    
    maxIters = 10000
    tStep = 0.1
    minObj = MinimizationAlgorithms(lneb)
    allPts, allForces, actions, strRep = \
        minObj.verlet_minimization(maxIters=maxIters,tStep=tStep)
        
    endTime = time.time()
    runTime = endTime - startTime
    analyticsDict = {"runTime":runTime}
        
    optimParamsDict = {"maxIters":maxIters,"tStep":tStep,\
                        "algorithm":strRep}
        
    allParamsDict = {"optimization":optimParamsDict,"lneb":lnebParamsDict,\
                      "interpolation":interpArgsDict,"analytics":analyticsDict}
    otpOutputsDict = {"allPts":allPts,"allVelocities":allVelocities,\
                      "allForces":allForces,"actions":actions}
    
    fName = "Runs/"+str(int(startTime))
    
    FileIO.dump_to_hdf5(fName,otpOutputsDict,allParamsDict)
        
    fig, ax = plt.subplots()
    ax.plot(actions)
    ax.set(xlabel="Iteration",ylabel="Action")
    fig.savefig("Runs/"+str(int(startTime))+"_Action.pdf")
    
    cbarRange = (-5,30)
    fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],dsets["PES"])
    ax.contour(dsets["Q20"],dsets["Q30"],dsets["PES"],levels=[constraintEneg],\
                colors=["black"])
    ax.plot(allPts[0,0,:],allPts[0,1,:],ls="-",marker="o")
    ax.plot(allPts[-1,0,:],allPts[-1,1,:],ls="-",marker="o")
    ax.set_ylim(0,30)
    ax.set(title=r"${}^{252}$U PES")
        
    fig.savefig("Runs/"+str(int(startTime))+".pdf")
    
    return None

def interp_mode_test():
    #Grid size for 252U
    nPtsCoarseGrid = (351,151)
    nPtsFineGrid = (501,501)
    
    ptRange = [(0,4),(-2,3.8)]
    
    xCoarse, yCoarse = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                    num=nPtsCoarseGrid[cIter]) for\
                        cIter in range(len(nPtsCoarseGrid))]
    xDense, yDense = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                  num=nPtsFineGrid[cIter]) for\
                      cIter in range(len(nPtsFineGrid))]
        
    coarseMesh = np.meshgrid(xCoarse,yCoarse)
    zCoarse = LepsPot(initialGrid=coarseMesh).potential(*coarseMesh)
    
    denseMesh = np.meshgrid(xDense,yDense)
    zDense = LepsPot(initialGrid=denseMesh).potential(*denseMesh)
    
    splineInterps = ["linear","cubic","quintic"]
    interpObjs = {}
    interpTimes = {}
    for interpMode in splineInterps:
        t0 = time.time()
        interpObjs[interpMode] = CustomInterp2d(xCoarse,yCoarse,zCoarse,kind=interpMode)
        t1 = time.time()
        interpTimes[interpMode] = t1 - t0
            
    densePreds = {}
    predictTimes = {}
    for interpMode in splineInterps:
        t0 = time.time()
        densePreds[interpMode] = interpObjs[interpMode].potential(*denseMesh)
        t1 = time.time()
        predictTimes[interpMode] = t1 - t0
        
    clipRange = (-5,-1)
    nLevels = 5
    fig, ax = plt.subplots(ncols=len(splineInterps),figsize=(12,4),sharex=True,sharey=True)
    for (modeIter, mode) in enumerate(splineInterps):
        plotDat = np.log10(np.abs((densePreds[mode]-zDense))).clip(*clipRange)
        cf = ax[modeIter].contourf(denseMesh[0],denseMesh[1],plotDat,\
                                    levels=np.linspace(clipRange[0],clipRange[1],nLevels))
        ax[modeIter].set(xlabel="x",title=mode.capitalize()+" Spline")
    
    plt.colorbar(cf,ax=ax[-1],label=r"Log${}_{10} | E_{Interp}-E_{Exact}|$")
    ax[0].set(ylabel="y")
    fig.savefig("LEPs_PES_Diff.pdf",bbox_inches="tight")
    
    absMeanDiffs = np.array([np.mean(np.abs(densePreds[interpMode] - \
                                              zDense)) for interpMode in splineInterps])
    meanFig, meanAx = plt.subplots()
    meanAx.scatter(np.arange(len(absMeanDiffs)),np.log10(absMeanDiffs))
    meanAx.set_xticks(np.arange(len(absMeanDiffs)))
    meanAx.set_xticklabels([s.capitalize() for s in splineInterps],rotation=45)
    meanAx.set(xlabel="Spline Mode",ylabel=r"log${}_{10}\langle | E_{Interp}-E_{Exact}|\rangle$")
    meanAx.set(title="Mean Energy Difference")
    meanFig.savefig("Mean_PES_Diff.pdf")
    
    timeFig, timeAx = plt.subplots()
    timeAx.scatter(np.arange(len(absMeanDiffs)),interpTimes.values(),label="Interpolating")
    timeAx.scatter(np.arange(len(absMeanDiffs)),predictTimes.values(),label="Predicting")
    timeAx.set_xticks(np.arange(len(absMeanDiffs)))
    timeAx.set_xticklabels([s.capitalize() for s in splineInterps],rotation=45)
    timeAx.legend()
    timeAx.set(xlabel="Spline Mode",ylabel="Time (s)",title="Total Run Times")
    timeFig.savefig("Interpolation_Time.pdf")
    
    return None

def gp_test():
    nPtsCoarseGrid = (351,151)
    nPtsFineGrid = (501,501)
    
    ptRange = [(0,4),(-2,3.8)]
    
    xCoarse, yCoarse = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                    num=nPtsCoarseGrid[cIter]) for\
                        cIter in range(len(nPtsCoarseGrid))]
    xDense, yDense = [np.linspace(ptRange[cIter][0],ptRange[cIter][1],\
                                  num=nPtsFineGrid[cIter]) for\
                      cIter in range(len(nPtsFineGrid))]
        
    coarseMesh = np.meshgrid(xCoarse,yCoarse)
    zCoarse = LepsPot(initialGrid=coarseMesh).potential(*coarseMesh)
    
    testingCutoff = 200
    denseMesh = np.meshgrid(xDense[:testingCutoff],yDense[:testingCutoff])
    zDense = LepsPot(initialGrid=denseMesh).potential(*denseMesh)
    
    locGP = LocalGPRegressor(xCoarse,yCoarse,gridPot=zCoarse)
    
    gpPred = locGP.potential(*denseMesh)
    
    Utilities.standard_pes(xDense[:testingCutoff],yDense[:testingCutoff],\
                           zDense[:testingCutoff,:testingCutoff]-gpPred)
    
    return None

def sylvester_otl_test():
    t0 = time.time()
    
    sp = SylvesterPot()
    zz = sp.potential(*sp.initialGrid)
    fig, ax = Utilities.standard_pes(*sp.initialGrid,zz,clipRange=(-0.1,10))
    
    initialPts = np.array([np.linspace(c[sp.minInds][0],c[sp.minInds][1],num=22) for\
                           c in sp.initialGrid])
    ax.plot(*initialPts,ls="-",marker=".",color="k")
    eConstraint = np.max(zz[sp.minInds])
    print(eConstraint)
    
    lneb = LineIntegralNeb(sp.potential,Utilities.const_mass(),initialPts,10,1,eConstraint,\
                           loggingLevel="output")
    minObj = MinimizationAlgorithms(lneb,loggingLevel="output")
    maxIters = 25
    allPts, allVelocities, allForces, actions = \
        minObj.verlet_minimization_v2(maxIters=maxIters,tStep=0.1)
        
    fName = "Default_Log"
    lneb.write_log(fName+".h5",overwrite=True)
    minObj.write_log(fName+".h5")
        
    minInd = np.argmin(actions)
    ax.plot(allPts[0,0],allPts[0,1,:],marker=".",ls="-")
    ax.plot(allPts[minInd,0],allPts[minInd,1,:],marker=".",ls="-")
    ax.plot(allPts[-1,0],allPts[-1,1,:],marker=".",ls="-")
        
    fig.savefig("Runs/Sylvester_PES.pdf")
        
    fig, ax = plt.subplots()
    ax.plot(np.arange(actions.shape[0]),actions)
    ax.set(xlabel="Iteration",ylabel="Action")
    fig.savefig("Runs/Sylvester_Action.pdf")
    
    t1 = time.time()
    print("Run time: %.3f" % (t1-t0))
    
    return None

#Actually important here lol
if __name__ == "__main__":
    sylvester_otl_test()