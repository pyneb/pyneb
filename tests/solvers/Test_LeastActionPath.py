from context import *

import unittest
import warnings

print("\nRunning "+os.path.relpath(__file__))

class _compute_tangents_(unittest.TestCase):
    def test_first_branch(self):
        def pot(coordsArr):
            return np.ones(3)
        
        nPts = 4
        nDims = 2
        
        lap = LeastActionPath(pot,nPts,nDims,logLevel=0)
        
        points = np.stack(2*(np.arange(4),)).T
        enegs = np.arange(4)
        
        tangents = lap._compute_tangents(points,enegs)
        
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
        
        lap = LeastActionPath(pot,nPts,nDims,logLevel=0)
        
        points = np.stack(2*(np.arange(4),)).T
        enegs = np.arange(3,-1,-1)
        
        tangents = lap._compute_tangents(points,enegs)
        
        correctTangents = np.zeros((nPts,nDims))
        correctTangents[1:3] = np.sqrt(2)/2
        
        #Use assert_allclose rather than assert_array_equal because of some quirk
        #in the calculation, which gives a difference of almost machine precision
        #(~10^(-16)) from the analytic correct answer, \sqrt{2}/2
        self.assertIsNone(np.testing.assert_allclose(tangents,correctTangents))
        
        return None
    
    def test_third_branch(self):
        def pot(coordsArr):
            return np.ones(3)
        
        nPts = 3
        nDims = 2
        
        lap = LeastActionPath(pot,nPts,nDims,logLevel=0)
        
        points = np.stack(2*(np.arange(3),)).T
        enegs = np.array([-1,-5,1])
        
        tangents = lap._compute_tangents(points,enegs)
        
        correctTangents = np.zeros((nPts,nDims))
        correctTangents[1] = np.sqrt(2)/2
        
        #Use assert_allclose rather than assert_array_equal because of some quirk
        #in the calculation, which gives a difference of almost machine precision
        #(~10^(-16)) from the analytic correct answer, \sqrt{2}/2
        self.assertIsNone(np.testing.assert_allclose(tangents,correctTangents))
        
        return None
    
    def test_last_branch(self):
        def pot(coordsArr):
            return np.ones(3)
        
        nPts = 3
        nDims = 2
        
        lap = LeastActionPath(pot,nPts,nDims,logLevel=0)
        
        points = np.stack(2*(np.arange(3),)).T
        #Maybe not obvious that this takes the last branch. Checked it by
        #putting a print statement under the last branch.
        enegs = np.array([-1,-5,-1])
        
        tangents = lap._compute_tangents(points,enegs)
        
        correctTangents = np.zeros((nPts,nDims))
        correctTangents[1] = np.sqrt(2)/2
        
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
        
        lap = LeastActionPath(pot,nPts,nDims,nebParams={"k":1},endpointSpringForce=endpointSpringForce,logLevel=0)
        tangents = np.zeros((nPts,nDims))
        tangents[1] = np.sqrt(2)/2
        
        springForce = lap._spring_force(points,tangents)
        
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
        
        lap = LeastActionPath(pot,nPts,nDims,nebParams={"k":1},endpointSpringForce=endpointSpringForce,logLevel=0)
        tangents = np.zeros((nPts,nDims))
        tangents[1] = np.sqrt(2)/2
        
        springForce = lap._spring_force(points,tangents)
        
        correctSpringForce = np.array([[1,1],[1,1],[-2,-2]])
        
        self.assertIsNone(np.testing.assert_allclose(springForce,correctSpringForce))
        
        return None
    
    def test_left_endpoint_force(self):
        def pot(coordsArr):
            return np.ones(3)
        
        endpointSpringForce = (True,False)
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3]),)).T
        
        lap = LeastActionPath(pot,nPts,nDims,nebParams={"k":1},endpointSpringForce=endpointSpringForce,logLevel=0)
        tangents = np.zeros((nPts,nDims))
        tangents[1] = np.sqrt(2)/2
        
        springForce = lap._spring_force(points,tangents)
        
        correctSpringForce = np.array([[1,1],[1,1],[0,0]])
        
        self.assertIsNone(np.testing.assert_allclose(springForce,correctSpringForce))
        
        return None
    
    def test_right_endpoint_force(self):
        def pot(coordsArr):
            return np.ones(3)
        
        endpointSpringForce = (False,True)
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3]),)).T
        
        lap = LeastActionPath(pot,nPts,nDims,nebParams={"k":1},endpointSpringForce=endpointSpringForce,logLevel=0)
        tangents = np.zeros((nPts,nDims))
        tangents[1] = np.sqrt(2)/2
        
        springForce = lap._spring_force(points,tangents)
        
        correctSpringForce = np.array([[0,0],[1,1],[-2,-2]])
        
        self.assertIsNone(np.testing.assert_allclose(springForce,correctSpringForce))
        
        return None
    
class compute_force_(unittest.TestCase):
    def test_correct_points(self):
        def pot(coordsArr):
            return coordsArr[:,0]**2 + coordsArr[:,1]**2
        
        nPts = 3
        nDims = 2
        
        points = np.stack(2*(np.array([0,1,3],dtype=float),)).T
        
        lap = LeastActionPath(pot,nPts,nDims,nebParams={"k":1,"kappa":2},logLevel=0)
        
        netForce = lap.compute_force(points)
        
        #Computed with Mathematica
        correctNetForce = np.array([[1,1],[1,1],[-25.455844,-25.455844]])
        
        self.assertIsNone(np.testing.assert_allclose(netForce,correctNetForce))
        
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()