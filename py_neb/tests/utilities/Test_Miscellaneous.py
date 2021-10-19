from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

"""
Tests:
    -mass_funcs_to_array_func

"""
class mass_funcs_to_array_func_(unittest.TestCase):
    def test_allowed_keys(self):
        def dummy_func(idx):
            def func_out(coords):
                return idx*coords[:,0]
            return func_out
        
        uniqueKeys = ["20","30"]
        dictOfFuncs = {"B2020":dummy_func(0),"B2030":dummy_func(1),\
                       "B3030":dummy_func(2)}
        
        func_out = mass_funcs_to_array_func(dictOfFuncs,uniqueKeys)
        coords = np.array([[0,0],[1,1],[2,2]])
        
        outputs = func_out(coords)
        correctOutputs = np.array([[[0,0],[0,0]],[[0,1],[1,2]],[[0,2],[2,4]]])
        
        self.assertIsNone(np.testing.assert_array_equal(outputs,correctOutputs))
            
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()