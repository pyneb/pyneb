#Suggested under https://docs.python-guide.org/writing/structure/
import os
import sys

pyNebDir = os.path.join(os.getcwd(),"..//../src")
if pyNebDir not in sys.path:
    sys.path.append(pyNebDir)
    # sys.path.insert(0,pyNebDir)
from pyneb import *
