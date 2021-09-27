#Suggested under https://docs.python-guide.org/writing/structure/
import os
import sys

pyNebDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..//.."))
if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
    
from py_neb import *