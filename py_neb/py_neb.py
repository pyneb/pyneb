import numpy as np
from scipy.ndimage import filters, morphology #For minimum finding
import h5py
import sys

"""
CONVENTIONS:
    -Paths should be of shape (nPoints, nDimensions)
    -Functions (e.g. a potential) that take in a single point should assume the
        first index of the array iterates over the points
    -Similarly, functions (e.g. the action) that take in many points should also
        assume the first index iterates over the points
"""

"""
Other functions we want:
    -Class for least action path and minimum energy path
    -Method(s) for finding starting points for a D-dimensional grid.
        -Would want this to be robust. Ideally, it would select the outer turning
            (D-1)-dimensional hypersurface. Simple idea is just to select *any* point
            with the same energy as the ground state, and see what happens. This brings
            up another point: is there a way to show we're at the outer turning line,
            without looking at the surface?
"""

"""
TODO:
    -Unclear how we want to handle errors (currently just using sys.exit)

"""

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
        D arrays of length k, for k minima found

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

def midpoint_grad(func,points,eps=10**(-8)):
    """
    Midpoint finite difference. Probably best if not used with actual DFT calculations,
        vs a forwards/reverse finite difference
    Assumes func only depends on a single point (vs the action, which depends on
         all of the points)
    """
    if len(points.shape) == 1:
        points = points.reshape((1,-1))
    nPoints, nDims = points.shape
    
    gradOut = np.zeros((nPoints,nDims))
    for ptIter in range(nPoints):
        for dimIter in range(nDims):
            step = np.zeros(nDims)
            step[dimIter] = 1
            
            forwardStep = points[ptIter] + eps/2*step
            backwardStep = points[ptIter] - eps/2*step
            
            forwardEval = func(forwardStep)
            backwardEval = func(backwardStep)
            
            gradOut[ptIter,dimIter] = (forwardEval - backwardEval)/eps
    
    return gradOut

def action(path,potential,masses=None):
    """
    Allowed masses:
        -Constant mass; set masses = None
        -Array of values; set masses to a numpy array of shape (nPoints, nDims, nDims)
        -A function; set masses to a function
    Allowed potential:
        -Array of values; set potential to a numpy array of shape (nPoints,)
        -A function; set masses to a function
        
    Computes action as
        $ S = \sum_{i=1}^{nPoints} \sqrt{2 E(x_i) M_{ab}(x_i) (x_i-x_{i-1})^a(x_i-x_{i-1})^b} $
    """
    nPoints, nDims = path.shape
    
    if masses is None:
        massArr = np.full((nPoints,nDims,nDims),np.identity(nDims))
    elif not isinstance(masses,np.ndarray):
        massArr = masses(path)
    else:
        massArr = masses
        
    massDim = (nPoints, nDims, nDims)
    if massArr.shape != massDim:
        raise ValueError("Dimension of massArr is "+str(massArr.shape)+\
                         "; required shape is "+str(massDim)+". See action function.")
    
    if not isinstance(potential,np.ndarray):
        potArr = potential(path)
    else:
        potArr = potential
        
    potShape = (nPoints,)
    if potArr.shape != potShape:
        raise ValueError("Dimension of potArr is "+str(potArr.shape)+\
                         "; required shape is "+str(potShape)+". See action function.")
        
    #Actual calculation
    actOut = 0
    for ptIter in range(1,nPoints):
        coordDiff = path[ptIter] - path[ptIter - 1]
        dist = np.dot(coordDiff,np.dot(massArr[ptIter],coordDiff)) #The M_{ab} dx^a dx^b bit
        actOut += np.sqrt(2*potArr[ptIter]*dist)
    
    return actOut, potArr, massArr

def forward_action_grad(path,potential,potentialOnPath,mass,massOnPath,\
                        target_func):
    """
    potential and mass are as allowed in "action" func; will let that do the error
    checking (for now...?)
    
    Takes forwards finite difference approx of any action-like function
    
    Does not return the gradient of the mass function, as that's not used elsewhere
    in the algorithm
    
    Maybe put this + action inside of LeastActionPath? not sure how we want to structure that part
    """
    eps = 10**(-8)
    
    gradOfPes = np.zeros(path.shape)
    gradOfAction = np.zeros(path.shape)
    
    nPts, nDims = path.shape
    
    actionOnPath, _, _ = target_func(path,potentialOnPath,massOnPath)
    
    for ptIter in range(self.nPts):
        for dimIter in range(self.nDims):
            steps = points.copy()
            steps[ptIter,dimIter] += eps
            actionAtStep, potAtStep, massAtStep = target_func(steps,potential,mass)
            
            gradOfPes[ptIter,dimIter] = (potAtStep[ptIter] - potentialOnPath[ptIter])/eps
            gradOfAction[ptIter,dimIter] = (actionAtStep - actionOnPath)/eps
    
    return gradOfAction, gradOfPes

class CustomLogging:
    """
    A note on parallelization:
        -If one runs multiple instances of the NEB solver concurrently, one
            should output all instances into separate folders *outside* of
            this program (e.g. in the submit script)
        -If one uses a parallelized energy evaluation (e.g. a DFT solver),
            one should be careful when updating the log. Probably, we can
            assume that parallel energy evaluations will have their own
            parallelized logging, and we can simply update the log here once
            we gather all of the energy outputs, without parallelizing anything
    """
    def __init__(self,loggingLevel):
        allowedLogLevels = [None,"output"]#"output" doesn't keep track of intermediate steps
        assert loggingLevel in allowedLogLevels
        self.loggingLevel = loggingLevel
        if self.loggingLevel == "output":
            self.logDict = {}
            self.outputNms = {}
            
            self.stringRep = self.__str__()+"_"+datetime.datetime.now().isoformat()
    
    def update_log(self,strRep,outputTuple,outputNmsTuple,isTuple=True):
        #If returning a single value, set isTuple -> False
        if self.loggingLevel is None:
            return None
        
        if self.loggingLevel == "output":
            if not isTuple:
                outputTuple = (outputTuple,)
                outputNmsTuple = (outputNmsTuple,)
            
            gpName = self.stringRep+"/"+strRep
            if gpName not in self.logDict:
                self.logDict[gpName] = []
                for t in outputTuple:
                    if isinstance(t,np.ndarray):
                        self.logDict[gpName].append(np.expand_dims(t,axis=0))
                    else:
                        self.logDict[gpName].append([t])
                self.outputNms[gpName] = outputNmsTuple
            else:
                assert len(outputTuple) == len(self.logDict[gpName])
                for (tIter,t) in enumerate(outputTuple):
                    if isinstance(t,np.ndarray):
                        self.logDict[gpName][tIter] = \
                            np.concatenate((self.logDict[gpName][tIter],np.expand_dims(t,axis=0)))
                    else:
                        self.logDict[gpName][tIter].append(t)
                        
        return None
    
    def write_log(self,fName,overwrite=False):
        #WARNING: probably doesn't handle anything that isn't a numpy array, although
            #that's almost all that I intend to log at the moment
        #WARNING: does not handle multiple of the same class instance
        if not hasattr(self,"logDict"):
            return None
        
        if not fName.startswith("Logs/"):
            fName = "Logs/"+fName
        os.makedirs("Logs",exist_ok=True)
        
        if (overwrite) and (os.path.isfile(fName)):
            os.remove(fName)
        
        h5File = h5py.File(fName,"a")
        for key in self.logDict.keys():
            splitKey = key.split("/")
            for (sIter,s) in enumerate(splitKey):
                subGp = "/".join(splitKey[:sIter+1])
                if not subGp in h5File:
                    h5File.create_group(subGp)
            for (oIter,outputNm) in enumerate(self.outputNms[key]):
                h5File[key].create_dataset(outputNm,data=self.logDict[key][oIter])
        
        h5File.close()
        
        return None
    

class LeastActionPath:
    """
    class documentation...?
    """
    def __init__(self,potential,endpointSpringForce,nPts,nDims,mass=None,\
                 target_func=action,target_func_grad=forward_action_grad,nebParams={}):
        """
        asdf

        Parameters
        ----------
        potential : Function
            To be called as potential(path). Is passed to "target_func".
        endpointSpringForce : Bool or tuple of bools
            If a single bool, behavior is applied to both endpoints. If is a tuple
            of bools, the first stands for the index 0 on the path; the second stands
            for the index -1 on the path. TODO: possibly allow for a complicated
            function that returns a bool?
        nPts : Int
            Number of points on the band, including endpoints.
        nDims : Int
            Number of dimensions of the collective coordinates. For instance,
            when working with (Q20,Q30), nDims = 2.
        mass : Function, optional
            To be called as mass(path). Is passed to "target_func". If mass == None,
            the collective inertia is the identity matrix. The default is None.
        target_func : Function, optional
            The approximation of the action integral. Should take as arguments
            (path, potential, mass). Should return (action, potentialAtPath, massesAtPath).
            The default is action.
        target_func_grad : Function, optional
            Approximate derivative of the action integral with respect to every point.
            Should take as arguments 
                (path, potentialFunc, potentialOnPath, massFunc, massOnPath, target_func),
            where target_func is the action integral approximation. Should return 
            (gradOfAction, gradOfPes). The default is forward_action_grad.
        nebParams : Dict, optional
            Keyword arguments for the nudged elastic band (NEB) method. Controls
            the spring force and the harmonic oscillator potential. Default
            parameters are controlled by a dictionary in the __init__ method.
            The default is {}.

        Returns
        -------
        None.

        """
        #TODO: consider not having NEB parameters as a dictionary. Could be confusing...?
        defaultNebParams = {"k":10,"kappa":20,"constraintEneg":0}
        for key in defaultNebParams.values():
            if key not in nebParams:
                nebParams[key] = defaultNebParams[key]
        
        for key in nebParams.keys():
            setattr(self,key,nebParams[key])
            
        if isinstance(endpointSpringForce,bool):
            endpointSpringForce = 2*(endpointSpringForce,)
        if not isinstance(endpointSpringForce,tuple):
            sys.exit("Err: unknown value "+str(endpointSpringForce)+\
                     " for endpointSpringForce")
        
        self.potential = potential
        self.mass = mass
        self.endpointSpringForce = endpointSpringForce
        self.nPts = nPts
        self.nDims = nDims
        self.target_func = target_func
        self.target_func_grad = target_func_grad
    
    def _compute_tangents(self,points,energies):
        """
        Here for testing sphinx autodoc

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.
        energies : TYPE
            DESCRIPTION.

        Returns
        -------
        tangents : TYPE
            DESCRIPTION.

        """
        tangents = np.zeros((self.nPts,self.nDims))
        
        #Range selected to exclude endpoints. Tangents on the endpoints do not
        #appear in the formulas.
        for ptIter in range(1,self.nPts-1):
            tp = points[ptIter+1] - points[ptIter]
            tm = points[ptIter] - points[ptIter-1]
            dVMax = np.max(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
            dVMin = np.min(np.absolute([energies[ptIter+1]-energies[ptIter],\
                                        energies[ptIter-1]-energies[ptIter]]))
                
            if (energies[ptIter+1] > energies[ptIter]) and \
                (energies[ptIter] > energies[ptIter-1]):
                tangents[ptIter] = tp
            elif (energies[ptIter+1] < energies[ptIter]) and \
                (energies[ptIter] < energies[ptIter-1]):
                tangents[ptIter] = tm
            elif energies[ptIter+1] > energies[ptIter-1]:
                tangents[ptIter] = tp*dVMax + tm*dVMin
            else:
                tangents[ptIter] = tp*dVMin + tm*dVMax
                
            #Normalizing vectors
            tangents[ptIter] = tangents[ptIter]/np.linalg.norm(tangents[ptIter])
        
        return tangents
    
    def _spring_force(self,points,tangents):
        springForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            forwardDist = np.linalg.norm(poinst[i+1] - points[i])
            backwardsDist = np.linalg.norm(points[i] - points[i-1])
            springForce[:,i] = self.k*(forwardDist - backwardsDist)*tangents[i]
            
        if self.endpointSpringForce[0]:
            springForce[0] = self.k*(points[1] - points[0])
        
        if self.endpointSpringForce[1]:
            springForce[-1] = -self.k*(points[self.nPts] - points[self.nPts-1])
        
        return springForce
    
    def compute_force(self,points):
        expectedShape = (self.nPoints,self.nDims)
        if points.shape != expectedShape:
            if (points.T).shape == expectedShape:
                points = points.T
            else:
                sys.exit("Err: points "+str(points)+\
                         " does not match expected shape in LeastActionPath")
        
        integVal, energies, masses = self.target_func(points,self.potential,self.mass)
        
        tangents = self._compute_tangents(points,energies)
        gradOfAction, gradOfPes = \
            self.target_func_grad(points,self.potential,energies,self.mass,masses,\
                                  self.target_func)
                
        negIntegGrad = -gradOfAction
        trueForce = -gradOfPes
        
        #Note: don't care about the tangents on the endpoints; they don't show up
        #in the net force
        perpForce = negIntegGrad - tangents*(np.array([np.dot(negIntegGrad[i],tangents[i]) \
                                                       for i in range(self.nPoints)]))
        springForce = self._spring_force(points,tangents)
        
        #Computing optimal tunneling path force
        netForce = np.zeros(points.shape)
        for i in range(1,self.nPts-1):
            netForce[i] = perpForce[i] + springForce[i]
            
        #TODO: error checking in case trueForce[0] == np.zeros(...)
        normForce = trueForce[0]/np.linalg.norm(trueForce[0])
        netForce[0] = springForce[0] - (np.dot(springForce[0],normForce)-\
                                        self.kappa*(energies[0]-self.constraintEneg))*normForce
            
        normForce = trueForce[-1]/np.linalg.norm(trueForce[-1])
        netForce[-1] = springForce[-1] - (np.dot(springForce[-1],normForce)-\
                                          self.kappa*(energies[-1]-self.constraintEneg))*normForce
        
        return netForce
    
# class Minimum_energy_path_NEB():


    