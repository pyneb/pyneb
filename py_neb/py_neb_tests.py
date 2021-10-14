from py_neb import *
# import py_neb
# import os
# print(os.getcwd())
import unittest
import numpy as np
import warnings
# print(np)

"""
Use module unittest (see https://docs.python.org/3/library/unittest.html#module-unittest)
    for unit tests. It's much simpler.
    
    
List of tests to add:
    -Test find_local_minimum, using a 3D polynomial with known minima
    -Test midpoint_grad with polynomial:
        *With a single point, to test reshaping
        *With many points, just to check for sure
    -action:
        *Function mass/potential with wrong size outputs?
    -forward_action_grad:
        *Test with different allowed options for potential and mass
    -LeastActionPath.__init__:
        *Error checking
        *Initializing with different parameter sets
    -LeastActionPath.compute_force:
        *Feed in allowed points, transposed points, and wrong-sized points
        *Use polynomial in some number of dimensions for potential, and compute
            gradient by hand
        
Tests added:
    ==========================================================================
    07/09/2021
    -NDInterpWithBoundary.__init__:
        *Test unallowed boundaryHandler
    -NDInterpWithBoundary._find_indices:
        *Test with one point, inside of the grid region
    ==========================================================================
"""
    
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
            forward_action_grad(path,potential,potentialOnPath,mass,massOnPath,\
                                action)
              
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

class NDInterpWithBoundary_init_(unittest.TestCase):
    def test_unallowed_boundary_handler(self):
        dummyHandler = "linear"
        with self.assertRaises(ValueError):
            g = NDInterpWithBoundary((1,1),None,boundaryHandler=dummyHandler)
            
            
        return None
    
class NDInterpWithBoundary_find_indices_(unittest.TestCase):
    def test_in_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        #NDInterpWithBoundary is weird. In the current implementation, *calling*
        #g(xi) transposes the points beforce calling g._find_indices. There, it
        #is expected that xi has shape (ndims,-), while calling g(xi) expects
        #xi to have shape (-,ndims)
        point = np.array([0.2,0.4]).reshape((2,1))
        
        g = NDInterpWithBoundary((x,y),zz,minVal=None)
        
        indices, normDistances = g._find_indices(point)
        
        correctIndices = 2*(np.array([10],dtype=int),)
        correctDistances = [np.array([0.4]),np.array([0.8])]
        
        self.assertEqual(indices,correctIndices)
        self.assertEqual(normDistances,correctDistances)
        
        return None

class NDInterpWithBoundary_evaluate_linear_(unittest.TestCase):
    def test_in_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        point = np.array([0.2,0.4]).reshape((2,1))
        
        g = NDInterpWithBoundary((x,y),zz,minVal=None)
        indices, normDist = g._find_indices(point)
        
        values = g._evaluate_linear(indices, normDist)
        
        #Verified via Mathematica
        correctVal = np.array([0.3])
        
        #correctVal is off from expected value by ~floating point precision
        self.assertIsNone(np.testing.assert_allclose(values,correctVal))
        return None
    
class GridInterpWithBoundaries_call_(unittest.TestCase):
    def test_in_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        #Reminder that when calling NDInterpWithBoundary, the points should
        #have their *first* dimension equal to the number of coordinates
        point = np.array([0.2,0.4])
        
        g = NDInterpWithBoundary((x,y),zz,minVal=None)
        values = g(point)
        
        #Since we're in the interpolation region, should be the same output
        #as evaluating g._evaluate_linear()
        correctValues = np.array([0.3])
        
        self.assertIsNone(np.testing.assert_allclose(values,correctValues))
        return None
    
    def test_out_of_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        #Reminder that when calling NDInterpWithBoundary, the points should
        #have their *first* dimension equal to the number of coordinates
        point = np.array([5.2,0.4])
        
        g = NDInterpWithBoundary((x,y),zz,minVal=None)
        values = g(point)
        
        #Computed in Mathematica
        correctVal = np.array([39.41149756])
        
        self.assertIsNone(np.testing.assert_allclose(values,correctVal))
        return None
    
    def test_in_and_out_of_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        #Reminder that when calling NDInterpWithBoundary, the points should
        #have their *first* dimension equal to the number of coordinates
        points = np.array([[5.2,0.4],[0.,0.1],[-1,3]])
        
        g = NDInterpWithBoundary((x,y),zz,minVal=None)
        values = g(points)
        
        correctVals = np.array([39.41149756,0.05,10])
        self.assertIsNone(np.testing.assert_allclose(values,correctVals))
        return None
    
class potential_test(unittest.TestCase):
    def test_potential_function (self):
        path = np.arange(4).reshape((2,2))
        def potential(path):
            return path[:,0]**2 + path[:,1]**2
        def auxFunc(path):
            return path[:,0]**3 + path[:,1]**3
        
        eneg, aux_eneg = potential_target_func(path,potential,auxFunc=auxFunc)
        
        correctEneg = np.array([1,13])
        correctAux = np.array([1,35])
        
        self.assertIsNone(np.testing.assert_array_equal(eneg,correctEneg))
        self.assertIsNone(np.testing.assert_array_equal(aux_eneg,correctAux))
        
        return None
    def test_wrong_potential_shape(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(30)**2
        auxFunc = np.arange(30)**3
        with self.assertRaises(TypeError):
            potential(path,potential,auxFunc)
        return None
    
class potential_grad(unittest.TestCase):
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
class VerletMinimization_velocity_verlet(unittest.TestCase):
    def test_single_step(self):
        def pot(coords):
            return coords[:,0]**2 + 2*coords[:,1]**2
        
        nPts = 3
        nDims = 2
        lap = LeastActionPath(pot,nPts,nDims)
        
        initialPoints = np.array([[0.,0.],[1.,2.],[2.,3]])
        
        minObj = VerletMinimization(lap,initialPoints)
        allPts, allVelocities, allForces = minObj.velocity_verlet(0.1, 1)
        
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
            
            
        self.assertIsNone(np.testing.assert_allclose(allPts,correctPts))
        self.assertIsNone(np.testing.assert_allclose(allVelocities,correctVelocities))
        self.assertIsNone(np.testing.assert_allclose(allForces,correctForces))
        
        return None
    
    def test_step_with_damping(self):
        def pot(coords):
            return coords[:,0]**2 + 2*coords[:,1]**2
        
        nPts = 3
        nDims = 2
        lap = LeastActionPath(pot,nPts,nDims)
        
        initialPoints = np.array([[0.,0.],[1.,2.],[2.,3]])
        
        minObj = VerletMinimization(lap,initialPoints)
        allPts, allVelocities, allForces = \
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
                
        self.assertIsNone(np.testing.assert_allclose(allPts,correctPts))
        self.assertIsNone(np.testing.assert_allclose(allVelocities,correctVelocities))
        self.assertIsNone(np.testing.assert_allclose(allForces,correctForces))
        
        return None

if __name__ == "__main__":
    #Suppresses warnings that I know are present
    warnings.simplefilter("ignore")
    unittest.main()
