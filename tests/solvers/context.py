#Suggested under https://docs.python-guide.org/writing/structure/
import os
import sys
    
try:
    from pyneb import *
except ModuleNotFoundError:
    pyNebDir = os.path.join(os.getcwd(),"..//..","src")
    if pyNebDir not in sys.path:
        sys.path.insert(0,pyNebDir)
    from pyneb import *
