from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class __init_(unittest.TestCase):
    #Does, in fact, run this class, despite starting with two underscores
    def test_valid_path_id_mass(self):
        def pot(path):
            return path[:,0]**2 + path[:,1]**2
        
        path = np.array([[0.,0.],[1.,1.],[3.,3.]])
        enegOnPath = pot(path)
        
        elv = EulerLagrangeVerification(path,enegOnPath,pot)
        
        correctXDot = np.array([[0.,0.],[2.,2.],[4.,4.]])
        correctXDotDot = np.array([[0.,0.],[4.,4.],[0.,0.]])
        self.assertIsNone(np.testing.assert_array_equal(elv.xDot,correctXDot))
        self.assertIsNone(np.testing.assert_array_equal(elv.xDotDot,correctXDotDot))
        
        return None
    
class _compare_lagrangian_id_mass_(unittest.TestCase):
    def test_flat_wrong_geodesic(self):
        #Potential is flat, to keep things simple
        def pot(path):
            return np.ones(path.shape[0])
        
        path = np.array([[0.,0.],[1.,1.],[3.,3.]])
        enegOnPath = pot(path)
        
        elv = EulerLagrangeVerification(path,enegOnPath,pot)
        
        elDiff = elv._compare_lagrangian_id_inertia()
        correctDiff = np.array([[-8.,-8.]])
        self.assertIsNone(np.testing.assert_array_equal(elDiff,correctDiff))
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()