from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class _compute_tangents_(unittest.TestCase):
     def test_first_branch(self):
        def pot(coordsArr):
            return np.ones(3)
        
        nPts = 4
        nDims = 2
        
        mep = MinimumEnergyPath(pot,nPts,nDims)
        
        points = np.stack(2*(np.arange(4),)).T
        enegs = np.arange(4)
        
        tangents = mep._compute_tangents(points,enegs)
        
        correctTangents = np.zeros((nPts,nDims))
        correctTangents[1:3] = np.sqrt(2)/2
        
        #Use assert_allclose rather than assert_array_equal because of some quirk
        #in the calculation, which gives a difference of almost machine precision
        #(~10^(-16)) from the analytic correct answer, \sqrt{2}/2
        self.assertIsNone(np.testing.assert_allclose(tangents,correctTangents))
        
        return None
    
     def test_second_branch(self):
        def pot(coordsArr):
            return np.ones(3)
        
        nPts = 4
        nDims = 2
        
        mep = MinimumEnergyPath(pot,nPts,nDims)
        
        points = np.stack(2*(np.arange(4),)).T
        enegs = np.arange(3,-1,-1)
        
        tangents = mep._compute_tangents(points,enegs)
        
        correctTangents = np.zeros((nPts,nDims))
        correctTangents[1:3] = np.sqrt(2)/2
        
        #Use assert_allclose rather than assert_array_equal because of some quirk
        #in the calculation, which gives a difference of almost machine precision
        #(~10^(-16)) from the analytic correct answer, \sqrt{2}/2
        self.assertIsNone(np.testing.assert_allclose(tangents,correctTangents))
        
        return None
    
class _spring_force_(unittest.TestCase):
    def test_no_endpoint_force(self):
        def pot(coordsArr):
            return np.ones(3)
        
        endpointSpringForce = False
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3]),)).T
        
        mep = MinimumEnergyPath(pot,nPts,nDims,nebParams={"k":1},endpointSpringForce=endpointSpringForce)
        tangents = np.zeros((nPts,nDims))
        tangents[1] = np.sqrt(2)/2
        
        springForce = mep._spring_force(points,tangents)
        
        correctSpringForce = np.zeros((nPts,nDims))
        correctSpringForce[1] = 1
        
        self.assertIsNone(np.testing.assert_allclose(springForce,correctSpringForce))
        
        return None
    
    def test_all_endpoint_force(self):
        def pot(coordsArr):
            return np.ones(3)
        
        endpointSpringForce = True
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3]),)).T
        
        mep = MinimumEnergyPath(pot,nPts,nDims,nebParams={"k":1},endpointSpringForce=endpointSpringForce)
        tangents = np.zeros((nPts,nDims))
        tangents[1] = np.sqrt(2)/2
        
        springForce = mep._spring_force(points,tangents)
        
        correctSpringForce = np.array([[1,1],[1,1],[-2,-2]])
        
        self.assertIsNone(np.testing.assert_allclose(springForce,correctSpringForce))
        
        return None
    
class compute_force_(unittest.TestCase):
    def test_correct_points_Aux(self):
        def real_pot(coordsArr):
            return coordsArr[:,0]**2 + coordsArr[:,1]**2
        def auxFunc(coordsArr):
            return coordsArr[:,0]**3 + coordsArr[:,1]**3
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3],dtype=float),)).T
        # use default target func and target func grad.
        mep = MinimumEnergyPath(real_pot,nPts,nDims,auxFunc = auxFunc,nebParams={"k":1,"kappa":2})
        
        netForce = mep.compute_force(points)
        #Computed by hand. the force turned out the same as no aux. Not expected.
        correctNetForce = np.array([[1,1],[1,1],[-25.455844,-25.455844]])
        
        self.assertIsNone(np.testing.assert_allclose(netForce,correctNetForce))
        
        return None
    
    def test_correct_points_noAux(self):
        def real_pot(coordsArr):
            return coordsArr[:,0]**2 + coordsArr[:,1]**2
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3],dtype=float),)).T
        
        mep = MinimumEnergyPath(real_pot,nPts,nDims,nebParams={"k":1,"kappa":2})
        
        netForce = mep.compute_force(points)
        
        #Computed by hand
        correctNetForce = np.array([[1,1],[1,1],[-25.455844,-25.455844]])
        
        self.assertIsNone(np.testing.assert_allclose(netForce,correctNetForce))
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()