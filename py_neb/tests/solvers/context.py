#Suggested under https://docs.python-guide.org/writing/structure/
import os
import sys

packageDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..//.."))
if packageDir not in sys.path:
    sys.path.insert(0,packageDir)
    
from py_neb import *