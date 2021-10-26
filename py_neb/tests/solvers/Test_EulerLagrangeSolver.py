from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class _solve_id_inertia_(unittest.TestCase):
    def test_flat_pes(self):
        def pot(path):
            return np.ones(path.shape[0])
        
        path = np.array([[0.,0.],[1.,1.],[3.,3.]])
        
        els = EulerLagrangeSolver(path,pot)
        sol = els._solve_id_inertia()
        
        tDense = np.linspace(0,1,30)
        dt = 1/29
        
        lineVals = 3*np.array([i*dt for i in range(30)])
        actualSol = np.vstack((lineVals,lineVals,3*np.ones((2,30))))
        
        self.assertIsNone(np.testing.assert_array_equal(sol.sol(tDense),actualSol))
        return None
    
class solve_(unittest.TestCase):
    def test_flat_pes_id_mass(self):
        def pot(path):
            return np.ones(path.shape[0])
        
        path = np.array([[0.,0.],[1.,1.],[3.,3.]])
        
        els = EulerLagrangeSolver(path,pot)
        sol = els.solve()
        
        tDense = np.linspace(0,1,30)
        dt = 1/29
        
        lineVals = 3*np.array([i*dt for i in range(30)])
        actualSol = np.vstack((lineVals,lineVals,3*np.ones((2,30))))
        
        self.assertIsNone(np.testing.assert_array_equal(sol.sol(tDense),actualSol))
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()