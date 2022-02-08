from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class VerletMinimization_velocity_verlet(unittest.TestCase):
    def test_single_step(self):
        def pot(coords):
            return coords[:,0]**2 + 2*coords[:,1]**2
        
        nPts = 3
        nDims = 2
        lap = LeastActionPath(pot,nPts,nDims,logLevel=0)
        
        initialPoints = np.array([[0.,0.],[1.,2.],[2.,3]])
        
        minObj = VerletMinimization(lap,initialPoints)
        minObj.velocity_verlet(0.1, 1)
        
        #Computed via Mathematica, trusting the output of the force
        #evaluations of LeastActionPath
        correctPts = \
            np.array([[[0,0],[1,2],[2,3]],\
                      [[0.,0.],\
                       [0.950776509,1.87488184],\
                       [-0.177103256,-3.23130977]],\
                      [[1.60027052e-03,-8.00135259e-04],\
                       [8.10927161e-01,1.33434266e+00],\
                       [1.42853672e-01,3.03650492e+00]]])
        correctVelocities = \
            np.array([[[0,0],\
                       [-0.328156608,-0.834121052],\
                       [-14.5140217,-41.5420651]],\
                      [[0.0106684701,-0.00533423506],\
                       [-1.00661367,-3.89071629],\
                       [2.13304619,41.7854312]]])
        correctForces = \
            np.array([[[0.,0.],\
                       [-3.28156608,-8.34121052],\
                       [-145.14021705,-415.42065114]],\
                      [[0.106684701,-0.0533423506 ],\
                       [-7.83759615,-30.2935116],\
                       [21.3304619,417.854312]]])
            
        self.assertIsNone(np.testing.assert_allclose(minObj.allPts,correctPts))
        self.assertIsNone(np.testing.assert_allclose(minObj.allVelocities,correctVelocities))
        self.assertIsNone(np.testing.assert_allclose(minObj.allForces,correctForces))
        
        return None
    
    def test_step_with_damping(self):
        def pot(coords):
            return coords[:,0]**2 + 2*coords[:,1]**2
        
        nPts = 3
        nDims = 2
        lap = LeastActionPath(pot,nPts,nDims,logLevel=0)
        
        initialPoints = np.array([[0.,0.],[1.,2.],[2.,3]])
        
        minObj = VerletMinimization(lap,initialPoints)
        minObj.velocity_verlet(0.1, 1, dampingParameter=1)
        
        #Computed via Mathematica, trusting the output of the force
        #evaluations of LeastActionPath
        correctPts = \
            np.array([[[0,0],[1,2],[2,3]],\
                      [[0.,0.],\
                       [0.950776509,1.87488184],\
                       [-0.177103256,-3.23130977]],\
                      [[1.60027052e-03,-8.00135259e-04],\
                       [8.15849510e-01,1.34685447e+00],\
                       [3.60563998e-01,3.65963590e+00]]])
        correctVelocities = \
            np.array([[[0,0],\
                       [-0.328156608,-0.834121052],\
                       [-14.5140217,-41.5420651]],\
                      [[1.06684701e-02,-5.33423506e-03],\
                       [-9.73798014e-01,-3.80730418e+00],\
                       [3.58444836e+00,4.59396378e+01]]])
        correctForces = \
            np.array([[[0.,0.],\
                       [-3.28156608,-8.34121052],\
                       [-145.14021705,-415.42065114]],\
                      [[0.106684701,-0.0533423506 ],\
                       [-7.83759615,-30.2935116],\
                       [21.3304619,417.854312]]])
                
        self.assertIsNone(np.testing.assert_allclose(minObj.allPts,correctPts))
        self.assertIsNone(np.testing.assert_allclose(minObj.allVelocities,correctVelocities))
        self.assertIsNone(np.testing.assert_allclose(minObj.allForces,correctForces))
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()