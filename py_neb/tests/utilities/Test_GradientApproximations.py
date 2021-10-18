from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class __init___(unittest.TestCase):
    def test_1(self):
        #At the moment, nothing is actually initialized in the __init__ method.
        #If anything there breaks, I guess this'll tell us, but for now it's mostly
        #placeholder code
        g = GradientApproximations()
        return None
    
class discrete_element_(unittest.TestCase):
    #TODO: tests
    def test_1(self):
        return None

class discrete_sqr_action_grad_(unittest.TestCase):
    #TODO: tests
    def test_1(self):
        return None
    
class discrete_action_grad_(unittest.TestCase):
    #TODO: tests
    def test_1(self):
        return None
    
class forward_action_grad_(unittest.TestCase):
    def test_correct_inputs(self):
        path = np.array([[0,0],[1,1],[2,2]],dtype=float)
        
        def potential(path):
            return path[:,0]**2 + path[:,1]**2
        
        def mass(path):
            return np.full((3,2,2),np.identity(2),dtype=float)
        
        potentialOnPath = potential(path)
        massOnPath = mass(path)
        
        gradOfAction, gradOfPes = \
            GradientApproximations().forward_action_grad(path,potential,potentialOnPath,mass,massOnPath,\
                                                         TargetFunctions.action)
              
        #Computed by hand/Mathematica
        correctGradOfPes = np.array([[0,0],[2,2],[4,4]],dtype=float)
        correctGradOfAction = np.stack(2*(np.array([-np.sqrt(2),-1.77636*10**(-7),\
                                                    3*np.sqrt(2)]),)).T
        
        #Heuristic absolute tolerance of 10**(-8) makes it pass the test. Makes
        #sense that gradient isn't more precise, as finite-difference step is
        #eps = 10**(-8)
        self.assertIsNone(np.testing.assert_allclose(gradOfPes,correctGradOfPes,\
                                                     atol=10**(-8)))
        self.assertIsNone(np.testing.assert_allclose(gradOfAction,correctGradOfAction,\
                                                     atol=10**(-8)))
        
        return None
    
class forward_action_component_grad_(unittest.TestCase):
    def test_none_mass(self):
        def potential(coordArr):
            if coordArr.ndim == 1:
                coordArr = coordArr.reshape((1,-1))
            return coordArr[:,0]**2 + coordArr[:,1]**2
        
        path = np.array([[1.,2,3],[1,2,3]]).T
        potOnPath = potential(path)
        
        g = GradientApproximations()
        gradOut, gradOfPes = \
            g.forward_action_component_grad(path,potential,potOnPath,None,None,\
                                            TargetFunctions.action)
                
        correctGradOut = 3.*np.sqrt(2)*np.array([[0.,0],[1,1],[0,0]])
        correctPesGrad = np.array([[2.,2],[4,4],[6,6]])
        
        self.assertIsNone(np.testing.assert_allclose(gradOut,correctGradOut))
        self.assertIsNone(np.testing.assert_allclose(gradOfPes,correctPesGrad))
        
        return None
    
    def test_with_mass(self):
        def potential(coordArr):
            if coordArr.ndim == 1:
                coordArr = coordArr.reshape((1,-1))
            return coordArr[:,0]**2 + coordArr[:,1]**2
        
        def mass(coordArr):
            if coordArr.ndim == 1:
                coordArr = coordArr.reshape((1,-1))
            return np.full((coordArr.shape[0],2,2),np.identity(2))
        
        path = np.array([[1.,2,3],[1,2,3]]).T
        potOnPath = potential(path)
        massOnPath = mass(path)
        
        g = GradientApproximations()
        gradOut, gradOfPes = \
            g.forward_action_component_grad(path,potential,potOnPath,mass,massOnPath,\
                                            TargetFunctions.action)
                
        correctGradOut = 3.*np.sqrt(2)*np.array([[0.,0],[1,1],[0,0]])
        correctPesGrad = np.array([[2.,2],[4,4],[6,6]])
        
        self.assertIsNone(np.testing.assert_allclose(gradOut,correctGradOut))
        self.assertIsNone(np.testing.assert_allclose(gradOfPes,correctPesGrad))
        
        return None
    
class potential_central_grad_(unittest.TestCase):
    def test_correct_outputs(self):
        path = np.array([[0,0],[1,1],[2,2]],dtype=float)
        
        def potential(path):
            return path[:,0]**2 + path[:,1]**2
        
        def auxFunc(path):
            return path[:,0]**3 + path[:,1]**3
        
        potentialOnPath = potential(path)
        
        gradOfPES, gradOfAux = potential_central_grad(path,potential,auxFunc)
        #Computed by hand
        correctGradOfPes = np.stack(np.array([[0,0],[2,2],[4,4]],dtype=float))
        correctGradOfAux = np.stack(np.array([[0,0],[3,3],[12,12]],dtype=float))
        #Heuristic absolute tolerance of 10**(-8) makes it pass the test. Makes
        #sense that gradient isn't more precise, as finite-difference step is
        #eps = 10**(-8)
        self.assertIsNone(np.testing.assert_allclose(gradOfPES,correctGradOfPes,\
                                                     atol=10**(-8)))
        self.assertIsNone(np.testing.assert_allclose(gradOfAux,correctGradOfAux,\
                                                     atol=10**(-8)))
        return None
    def test_correct_outputs_no_aux(self):
        path = np.array([[0,0],[1,1],[2,2]],dtype=float)
        
        def potential(path):
            return path[:,0]**2 + path[:,1]**2
        
        potentialOnPath = potential(path)
        
        gradOfPES, gradOfAux = potential_central_grad(path,potential,auxFunc=None)
        #Computed by hand/Mathematica
        correctGradOfPes = np.array([[0,0],[2,2],[4,4]],dtype=float)
        correctGradOfAux = None
        
        #Heuristic absolute tolerance of 10**(-8) makes it pass the test. Makes
        #sense that gradient isn't more precise, as finite-difference step is
        #eps = 10**(-8)
        self.assertIsNone(np.testing.assert_allclose(gradOfPES,correctGradOfPes,\
                                                     atol=10**(-8)))
        self.assertIsNone(gradOfAux)
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()