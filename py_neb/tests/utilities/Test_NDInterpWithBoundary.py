from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class __init___(unittest.TestCase):
    def test_2d_init_same_symmExtend_everywhere(self):
        gridPoints = (np.arange(4),np.arange(6))
        gridVals = np.random.rand(6,4)
        g = NDInterpWithBoundary(gridPoints,gridVals,symmExtend=True)
        
        return None
    
    def test_3d_init(self):
        gridPoints = (np.arange(4),np.arange(6),np.arange(30))
        gridVals = np.random.rand(6,4,30)
        g = NDInterpWithBoundary(gridPoints,gridVals)
        
        return None
    
    def test_unallowed_boundary_handler(self):
        dummyHandler = "linear"
        with self.assertRaises(ValueError):
            g = NDInterpWithBoundary((1,1),None,boundaryHandler=dummyHandler)
            
        return None
    
    def test_unallowed_symm_extend(self):
        gridPoints = (np.arange(4),np.arange(6))
        gridVals = np.random.rand(6,4)
        symmExtend = np.zeros(3,dtype=bool)
        with self.assertRaises(ValueError):
            g = NDInterpWithBoundary(gridPoints,gridVals,symmExtend=symmExtend)
            
        return None
    
    def test_wrong_gridVals_shape(self):
        gridPoints = (np.arange(4),np.arange(6))
        gridVals = np.random.rand(6,5)
        with self.assertRaises(ValueError):
            g = NDInterpWithBoundary(gridPoints,gridVals)
            
        return None
    
    def test_out_of_order_points(self):
        x = np.array([1.,2.,4,3])
        gridPoints = (x,np.arange(12))
        gridVals = np.arange(12*4).reshape((12,4))
        
        with self.assertRaises(ValueError):
            g = NDInterpWithBoundary(gridPoints,gridVals)
            
        return None
    
    def test_1d_init(self):
        gridPoints = (np.arange(12),)
        gridVals = np.arange(12)
        
        with self.assertRaises(NotImplementedError):
            g = NDInterpWithBoundary(gridPoints,gridVals,symmExtend=False)
            
        return None
    
class _call_2d_(unittest.TestCase):
    def test_call(self):
        gridPoints = (np.arange(4),np.arange(6))
        mesh = np.meshgrid(*gridPoints)
        gridVals = mesh[0]**2 + mesh[1]**2
        
        g = NDInterpWithBoundary(gridPoints,gridVals)
        
        point = np.array([0.,2.3])
        val = g(point)
        
        #For a polynomial with degree less than the degree of the spline, the
        #spline interpolator will actually be equal to the polynomial.
        correctVal = np.array([5.29])
        
        #Floating point nonsense
        self.assertIsNone(np.testing.assert_allclose(val,correctVal))
        
        return None
    
class _find_indices_(unittest.TestCase):
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
        
        g = NDInterpWithBoundary((x,y),zz)
        
        indices, normDistances = g._find_indices(point)
        
        correctIndices = 2*(np.array([10],dtype=int),)
        correctDistances = np.array([[0.4],[0.8]])
        
        self.assertEqual(indices,correctIndices)
        self.assertIsNone(np.testing.assert_array_equal(normDistances,correctDistances))
        
        return None
    
    def test_out_of_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        #NDInterpWithBoundary is weird. In the current implementation, *calling*
        #g(xi) transposes the points beforce calling g._find_indices. There, it
        #is expected that xi has shape (ndims,-), while calling g(xi) expects
        #xi to have shape (-,ndims)
        point = np.array([5.2,0.4]).reshape((2,1))
        
        g = NDInterpWithBoundary((x,y),zz)
        
        indices, normDistances = g._find_indices(point)
        
        correctIndices = (np.array([19],dtype=int),np.array([10],dtype=int))
        correctDistances = np.array([[1.4],[0.8]])
        
        self.assertEqual(indices,correctIndices)
        self.assertIsNone(np.testing.assert_allclose(normDistances,correctDistances))
        
        return None

class _exp_boundary_handler_(unittest.TestCase):
    def test_one_bad_coord(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        g = NDInterpWithBoundary((x,y),zz)
        
        point = np.array([5.2,0.1])
        isInBounds = np.ones((2,2),dtype=bool)
        isInBounds[1,0] = False
        
        scaledVal = g._exp_boundary_handler(point,isInBounds)
        
        #Polynomials are guaranteed to be evaluated correctly with this interpolator
        #(test is in 2D). Tested in Mathematica
        correctVal = 39.114347381543666
        
        self.assertEqual(scaledVal,correctVal)
        
        return None
        
class _call_nd_(unittest.TestCase):
    def test_in_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        point = np.array([0.2,0.4]).reshape((2,1))
        
        g = NDInterpWithBoundary((x,y),zz)
        
        value = g._call_nd(point)
        
        #Verified via Mathematica
        correctVal = 0.3
        
        #correctVal is off from expected value by ~floating point precision
        self.assertAlmostEqual(value,correctVal)
        
        return None
    
class __call___(unittest.TestCase):
    def test_in_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        point = np.array([0.2,0.4])
        
        g = NDInterpWithBoundary((x,y),zz)
        values = g(point)
        
        #Since we're in the interpolation region, should be the interpolated
        #value; for spline interpolation of polynomials, is the true value
        correctValues = np.array([0.2])
        
        self.assertIsNone(np.testing.assert_allclose(values,correctValues))
        
        return None
    
    def test_out_of_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        point = np.array([5.2,0.4])
        
        g = NDInterpWithBoundary((x,y),zz)
        values = g(point)
        
        #Computed in Mathematica
        correctVal = np.array([39.34893963])
        
        self.assertIsNone(np.testing.assert_allclose(values,correctVal))
        
        return None
    
    def test_in_and_out_of_bounds(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        points = np.array([[5.2,0.4],[0.,0.1],[-1,3]])
        
        g = NDInterpWithBoundary((x,y),zz)
        values = g(points)
        
        correctVals = np.array([39.34893963,0.01,10])
        self.assertIsNone(np.testing.assert_allclose(values,correctVals))
        
        return None
    
    def test_wrong_points_shape(self):
        x = np.arange(-5,5.5,0.5)
        y = x.copy()
        
        xx, yy = np.meshgrid(x,y)
        zz = xx**2 + yy**2
        
        g = NDInterpWithBoundary((x,y),zz)
        
        with self.assertRaises(ValueError):
            point = np.arange(8)
            g(point)
            
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()