import h5py

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
    def __init__(self,logLevel=1):
        self.logLevel = logLevel
        self.initTime = datetime.datetime.now().isoformat()
        
    def _level_1(self):
        h5File = h5py.File(self.initTime+".djk","w")
        
        
        h5File.close()
        return None
    
