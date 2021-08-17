from py_neb import *

"""
CONVENTIONs:
    -Put tests for every function in their own class, as static methods, for
        namespace purposes
    -Name the class the name of the function that's being validated, ending with
        an underscore
    -Name the functions something reasonably descriptive
    -Have not yet decided naming conventions for class tests...
    -Have not decided how to handle automatic testing (should tests return a
        boolean, for instance)
"""


class action_:
    @staticmethod
    def constant_mass_array_potential():
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        
        act = action(path,potential)
        
        correctAction = 40
        print(act==correctAction)
        
        return None
    
    @staticmethod
    def constant_mass_function_potential():
        return None
    
action_.constant_mass_array_potential()
