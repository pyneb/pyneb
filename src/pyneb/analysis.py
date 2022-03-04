import numpy as np
import matplotlib.pyplot as plt

from utilities import *
from fileio import *

"""
Plots to make:
    -Action vs iterations
    -Path(s) on PES/inertia components
    -PES/inertia along path (1D)
    -Raw PES/inertia
    -Cumulative action?

Want to just feed in a list of log files, and make the plots from there. Can
also be compatible with feeding in a class instance, in e.g. LAP cases
"""

def action_vs_iteration(inputObj,enegFunc,massFunc=None,
                        target_func=TargetFunctions.action,interp=False,
                        nImages=None,interpKWargs={},figName=None):
    fig, ax = plt.subplots()
    
    if isinstance(inputObj,str):
        inputObj = LoadForceLogger(inputObj)
    
    return fig, ax