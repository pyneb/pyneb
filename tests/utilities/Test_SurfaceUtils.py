from context import *

import unittest
import warnings

print("\nRunning "+os.path.relpath(__file__))

class SaneEqualityArray(np.ndarray):
    """
    Taken from https://stackoverflow.com/a/14276901
    
    The way unittest checks equality is (apparently) by referencing variables'
    __eq__ method, but numpy arrays don't have that (and there is apparently a
    good reason for that). This subclasses that to fix the issue
    """
    def __eq__(self, other):
        return (isinstance(other, np.ndarray) and
                self.shape == other.shape and
                np.allclose(self, other))

class find_all_local_minimum_(unittest.TestCase):
    def test_sin_func(self):
        def func(arr):
            return np.sin(np.pi*arr[:,0])*np.sin(2*np.pi*arr[:,1])
        
        xUn = np.arange(0,2.25,0.25)
        yUn = np.arange(0,3.125,0.125)
        
        coordMeshTuple = np.meshgrid(xUn,yUn)
        #It must be ordered in this way to get the correct reshaped array
        zz = func(np.array([[x,y] for y in yUn for x in xUn])).reshape((len(yUn),len(xUn)))
        
        minInds = SurfaceUtils.find_all_local_minimum(zz)
        
        # fig, ax = plt.subplots()
        # ax.contourf(*coordMeshTuple,zz)
        # ax.scatter(coordMeshTuple[0][minInds],coordMeshTuple[1][minInds],\
        #            color="r",marker="x")
            
        #The way the minimum finder works, is it checks along the cardinal directions.
        #If all points are greater than or equal to the current value, the index
        #is returned as a minimum. So, in our test case, we also select out the
        #saddle points at x=1, y = 0,0.5,1,...
        correctXInds = np.array([0, 1, 2, 3, 4, 8, 0, 0, 6, 0, 0, 4, 8, 8, 2, 8, 8, 0, 4, 8, 0, 0,
                                 6, 0, 0, 4, 8, 8, 2, 8, 8, 0, 4, 8, 0, 0, 6, 0, 0, 4, 8, 8, 2, 8,
                                 8, 0, 4, 5, 6, 7, 8])
        correctXInds = SaneEqualityArray(correctXInds.shape,int,correctXInds)
        correctYInds = np.array([ 0,  0,  0,  0,  0,  0,  1,  2,  2,  3,  4,  4,  4,  5,  6,  6,  7,
                                 8,  8,  8,  9, 10, 10, 11, 12, 12, 12, 13, 14, 14, 15, 16, 16, 16,
                                 17, 18, 18, 19, 20, 20, 20, 21, 22, 22, 23, 24, 24, 24, 24, 24, 24])
        correctYInds = SaneEqualityArray(correctYInds.shape,int,correctYInds)
        correctInds = (correctYInds,correctXInds)
        
        self.assertTupleEqual(correctInds,minInds)
        
        return None
    
class find_local_minimum_(unittest.TestCase):
    def test_sin_func(self):
        def func(arr):
            return np.sin(np.pi*arr[:,0])*np.sin(2*np.pi*arr[:,1])
        
        xUn = np.arange(0,2.25,0.25)
        yUn = np.arange(0,3.125,0.125)
        
        coordMeshTuple = np.meshgrid(xUn,yUn)
        #It must be ordered in this way to get the correct reshaped array
        zz = func(np.array([[x,y] for y in yUn for x in xUn])).reshape((len(yUn),len(xUn)))
        
        minInds = SurfaceUtils.find_local_minimum(zz)
        
        correctMinInds = (6,2)
        
        self.assertTupleEqual(correctMinInds,minInds)
        # fig, ax = plt.subplots()
        # ax.contourf(*coordMeshTuple,zz)
        # ax.scatter(coordMeshTuple[0][minInds],coordMeshTuple[1][minInds],\
        #             color="r",marker="x")
        return None
    
    def test_wrong_search_perc_length(self):
        arr = np.ones((5,7))
        searchPerc = [0.1,0.2,0.3]
        with self.assertRaises(TypeError):
            SurfaceUtils.find_local_minimum(arr,searchPerc=searchPerc)
            
        return None
    
    def test_wrong_too_small_searchPerc(self):
        arr = np.ones((5,7))
        searchPerc = [0.1,-1]
        with self.assertRaises(ValueError):
            SurfaceUtils.find_local_minimum(arr,searchPerc=searchPerc)
            
        return None
    
    def test_wrong_too_large_searchPerc(self):
        arr = np.ones((5,7))
        searchPerc = [0.1,1.2]
        with self.assertRaises(ValueError):
            SurfaceUtils.find_local_minimum(arr,searchPerc=searchPerc)
            
        return None
    
class find_approximate_contours_(unittest.TestCase):
    def test_1d(self):
        dummyTuple = (np.arange(12),)
        zz = None
        with self.assertRaises(NotImplementedError):
            SurfaceUtils.find_approximate_contours(dummyTuple,zz)
        
        return None
    
    #TODO: tests that'll actually pass (can't think of any easy ones right now)
    
class round_points_to_grid_(unittest.TestCase):
    def test_2d_valid_input(self):
        x = np.arange(4)
        y = np.arange(2)
        coordMeshTuple = np.meshgrid(x,y)
        
        ptsToCheck = np.array([[3.,2.],[2.5,1],[1.3,-1]])
        
        indsOut, gridValsOut = SurfaceUtils.round_points_to_grid(coordMeshTuple,ptsToCheck)
        
        correctInds = np.array([[1,3],[1,2],[0,1]],dtype=int)
        correctVals = np.array([[3,1],[2,1],[1,0]],dtype=float)
        
        self.assertIsNone(np.testing.assert_array_equal(correctInds,indsOut))
        self.assertIsNone(np.testing.assert_array_equal(correctVals,gridValsOut))
        
        return None
    
    def test_1d(self):
        dummyTuple = (np.arange(12),)
        ptsArr = None
        with self.assertRaises(TypeError):
            SurfaceUtils.round_points_to_grid(dummyTuple,ptsArr)
            
        return None
    
    def test_wrong_shape_points(self):
        coordMeshTuple = np.meshgrid(np.arange(4),np.arange(3))
        ptsArr = np.arange(30).reshape((10,3))
        with self.assertRaises(ValueError):
            SurfaceUtils.round_points_to_grid(coordMeshTuple,ptsArr)
            
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
            SurfaceUtils.find_endpoints_on_grid(coordMeshTuple,zz,returnAllPoints=True,
                                                returnIndices=True)
        
        lineEndpoints = np.array([[-1.9,i] for i in y])
        #Validated by looking at contours (from contourf) and how they round
        #to the grid
        otherEndpoints = np.array([[1.9,0.],[2.,-0.1],[2.,0.]])
        correctEndpoints = np.concatenate((lineEndpoints,otherEndpoints))
        
        lineIndices = np.array([[i,31] for i in range(len(y))],dtype=int)
        otherIndices = np.array([[20,69],[19,70],[20,70]],dtype=int)
        
        correctIndices = np.concatenate((lineIndices,otherIndices))
        
        self.assertIsNone(np.testing.assert_allclose(allowedEndpoints,correctEndpoints,atol=10**(-13)))
        #Sometimes the rows end up out of order
        self.assertIsNone(np.testing.assert_array_equal(np.sort(endpointIndices,axis=0),\
                                                        np.sort(correctIndices,axis=0)))
        
        # #Left in because I kept deleting it and re-adding it
        # fig, ax = plt.subplots()
        # ax.contourf(*coordMeshTuple,zz)
        # ax.scatter(allowedEndpoints[:,0],allowedEndpoints[:,1],marker="x",color="red")
        return None
    
    def test_2d_dont_return_all_points(self):
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
            SurfaceUtils.find_endpoints_on_grid(coordMeshTuple,zz,returnIndices=True)
        
        #Same test as test_2d_return_all_points, but the only points to be returned
        #are the line, as those are the largest set
        correctEndpoints = np.array([[-1.9,i] for i in y])
        correctIndices = np.array([[i,31] for i in range(len(y))],dtype=int)
        
        self.assertIsNone(np.testing.assert_allclose(allowedEndpoints,correctEndpoints,atol=10**(-13)))
        self.assertIsNone(np.testing.assert_array_equal(endpointIndices,correctIndices))
        
        # #Left in because I kept deleting it and re-adding it
        # fig, ax = plt.subplots()
        # ax.contourf(*coordMeshTuple,zz)
        # ax.scatter(allowedEndpoints[:,0],allowedEndpoints[:,1],marker="x",color="red")
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()