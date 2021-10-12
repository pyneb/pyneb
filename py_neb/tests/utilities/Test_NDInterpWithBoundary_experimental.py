from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class NDInterpWithBoundary_init_(unittest.TestCase):
    def test_unallowed_boundary_handler(self):
        dummyHandler = "linear"
        with self.assertRaises(ValueError):
            g = NDInterpWithBoundary_experimental((1,1),None,boundaryHandler=dummyHandler)
            
            
        return None
    
# class NDInterpWithBoundary_find_indices_(unittest.TestCase):
#     def test_in_bounds(self):
#         x = np.arange(-5,5.5,0.5)
#         y = x.copy()
        
#         xx, yy = np.meshgrid(x,y)
#         zz = xx**2 + yy**2
        
#         #NDInterpWithBoundary is weird. In the current implementation, *calling*
#         #g(xi) transposes the points beforce calling g._find_indices. There, it
#         #is expected that xi has shape (ndims,-), while calling g(xi) expects
#         #xi to have shape (-,ndims)
#         point = np.array([0.2,0.4]).reshape((2,1))
        
#         g = NDInterpWithBoundary((x,y),zz,minVal=None)
        
#         indices, normDistances = g._find_indices(point)
        
#         correctIndices = 2*(np.array([10],dtype=int),)
#         correctDistances = [np.array([0.4]),np.array([0.8])]
        
#         self.assertEqual(indices,correctIndices)
#         self.assertEqual(normDistances,correctDistances)
        
#         return None

# class NDInterpWithBoundary_evaluate_linear_(unittest.TestCase):
#     def test_in_bounds(self):
#         x = np.arange(-5,5.5,0.5)
#         y = x.copy()
        
#         xx, yy = np.meshgrid(x,y)
#         zz = xx**2 + yy**2
        
#         point = np.array([0.2,0.4]).reshape((2,1))
        
#         g = NDInterpWithBoundary((x,y),zz,minVal=None)
#         indices, normDist = g._find_indices(point)
        
#         values = g._evaluate_linear(indices, normDist)
        
#         #Verified via Mathematica
#         correctVal = np.array([0.3])
        
#         #correctVal is off from expected value by ~floating point precision
#         self.assertIsNone(np.testing.assert_allclose(values,correctVal))
#         return None
    
# class GridInterpWithBoundaries_call_(unittest.TestCase):
#     def test_in_bounds(self):
#         x = np.arange(-5,5.5,0.5)
#         y = x.copy()
        
#         xx, yy = np.meshgrid(x,y)
#         zz = xx**2 + yy**2
        
#         #Reminder that when calling NDInterpWithBoundary, the points should
#         #have their *first* dimension equal to the number of coordinates
#         point = np.array([0.2,0.4])
        
#         g = NDInterpWithBoundary((x,y),zz,minVal=None)
#         values = g(point)
        
#         #Since we're in the interpolation region, should be the same output
#         #as evaluating g._evaluate_linear()
#         correctValues = np.array([0.3])
        
#         self.assertIsNone(np.testing.assert_allclose(values,correctValues))
#         return None
    
#     def test_out_of_bounds(self):
#         x = np.arange(-5,5.5,0.5)
#         y = x.copy()
        
#         xx, yy = np.meshgrid(x,y)
#         zz = xx**2 + yy**2
        
#         #Reminder that when calling NDInterpWithBoundary, the points should
#         #have their *first* dimension equal to the number of coordinates
#         point = np.array([5.2,0.4])
        
#         g = NDInterpWithBoundary((x,y),zz,minVal=None)
#         values = g(point)
        
#         #Computed in Mathematica
#         correctVal = np.array([39.41149756])
        
#         self.assertIsNone(np.testing.assert_allclose(values,correctVal))
#         return None
    
#     def test_in_and_out_of_bounds(self):
#         x = np.arange(-5,5.5,0.5)
#         y = x.copy()
        
#         xx, yy = np.meshgrid(x,y)
#         zz = xx**2 + yy**2
        
#         #Reminder that when calling NDInterpWithBoundary, the points should
#         #have their *first* dimension equal to the number of coordinates
#         points = np.array([[5.2,0.4],[0.,0.1],[-1,3]])
        
#         g = NDInterpWithBoundary((x,y),zz,minVal=None)
#         values = g(points)
        
#         correctVals = np.array([39.41149756,0.05,10])
#         self.assertIsNone(np.testing.assert_allclose(values,correctVals))
#         return None
    
if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    unittest.main()