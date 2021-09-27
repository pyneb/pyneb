from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

"""
Tests:
    -find_local_minimum
    -find_approximate_contours
    -round_points_to_grid
    -find_endpoints_on_grid

"""

class find_local_minimum_(unittest.TestCase):
    def test_todo(self):
        #TODO add test
        return None
    
class find_approximate_contours_(unittest.TestCase):
    def test_todo(self):
        #TODO add test
        return None
    
class round_points_to_grid_(unittest.TestCase):
    def test_2d_valid_input(self):
        x = np.arange(4)
        y = np.arange(2)
        coordMeshTuple = np.meshgrid(x,y)
        
        ptsToCheck = np.array([[3.,2.],[2.5,1],[1.3,-1]])
        
        indsOut, gridValsOut = round_points_to_grid(coordMeshTuple,ptsToCheck)
        
        correctInds = np.array([[3,1],[2,1],[1,0]],dtype=int)
        correctVals = np.array([[3,1],[2,1],[1,0]],dtype=float)
        
        self.assertIsNone(np.testing.assert_array_equal(correctInds,indsOut))
        self.assertIsNone(np.testing.assert_array_equal(correctVals,gridValsOut))
        
        return None

class find_endpoints_on_grid_(unittest.TestCase):
    def test_2d_return_all_points(self):
        #NOTE: corrected this test on Sep. 27, 2021, due to error in handling
        #points on the grid
        
        #Don't need anything more complicated than a function taking in a meshgrid
        def dummy_func(meshGrid):
            x, y = meshGrid
            return x*(1-2*np.exp(-((x-2)**2+y**2)/0.2)) + 1.9
                
        x = np.arange(-5,5.1,0.1)
        y = np.arange(-2,2.1,0.1)
        
        coordMeshTuple = np.meshgrid(x,y)
        zz = dummy_func(coordMeshTuple)
        
        initialPoint = np.array([2.,0.])
        
        allowedEndpoints, endpointIndices = \
            find_endpoints_on_grid(coordMeshTuple,zz,returnAllPoints=True)
        
        lineEndpoints = np.array([[-1.9,i] for i in y])
        #Validated by looking at contours (from contourf) and how they round
        #to the grid
        otherEndpoints = np.array([[1.9,0.],[2.,-0.1],[2.,0.]])
        correctEndpoints = np.concatenate((lineEndpoints,otherEndpoints))
        
        lineIndices = np.array([[31,i] for i in range(len(y))],dtype=int)
        otherIndices = np.array([[69,20],[70,19],[70,20]],dtype=int)
        correctIndices = np.concatenate((lineIndices,otherIndices))
        
        self.assertIsNone(np.testing.assert_allclose(allowedEndpoints,correctEndpoints,atol=10**(-13)))
        self.assertIsNone(np.testing.assert_array_equal(endpointIndices,correctIndices))
        
        # #Left in because I kept deleting it and re-adding it
        # fig, ax = plt.subplots()
        # ax.contourf(*coordMeshTuple,zz)
        # ax.scatter(allowedEndpoints[:,0],allowedEndpoints[:,1],marker="x",color="red")
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()