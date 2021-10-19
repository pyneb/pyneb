from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

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

class _construct_path_dict_(unittest.TestCase):
    def test_3d_grid(self):
        def dist_func(coords,enegs,masses):
            return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
        x1 = np.array([0,1])
        x2 = np.array([0,2])
        x3 = np.array([1,3])
        
        coordMeshTuple = np.meshgrid(x1,x2,x3)
        zz = np.arange(8).reshape((2,2,2))
        initialPoint = np.array([0,0,1])
        finalPoint = np.array([1,2,3])
        
        dijkstra = Dijkstra(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                            allowedEndpoints=finalPoint)
        dist, visitDict, endptList = \
            dijkstra._construct_path_dict()
        #Computed ~mostly by hand (did not check that these are the
        #shortest distances b/c I'm lazy)
        correctDistances = np.array([[[0.,2.],[2,5]],\
                                     [[8,12.],[6*5**0.5,2+7*5**0.5]]])
        correctVisitDict = \
            {(1,1,1):(0,0,1),(1,1,0):(0,0,0),(1,0,1):(0,0,1),(1,0,0):(0,0,0),\
             (0,1,1):(0,0,1),(0,1,0):(0,0,0),(0,0,1):(0,0,0)}
            
        self.assertIsNone(np.testing.assert_array_equal(correctDistances,dist.data))
        self.assertDictEqual(visitDict,correctVisitDict)
        self.assertListEqual(endptList,[])
        
        return None
    
class _get_paths_(unittest.TestCase):
    def test_3d_grid_two_endpoints(self):
        x1 = np.array([0,1])
        x2 = np.array([0,2])
        x3 = np.array([1,3])
        
        coordMeshTuple = np.meshgrid(x1,x2,x3)
        zz = np.arange(8).reshape((2,2,2))
        initialPoint = np.array([0,0,1])
        finalPoint = np.array([[1,2,3],[1,2,1]])
        
        dijkstra = Dijkstra(initialPoint,coordMeshTuple,zz,\
                            allowedEndpoints=finalPoint)
            
        #Taken from _construct_path_dict_.test_3d_grid
        visitDict = \
            {(1,1,1):(0,0,1),(1,1,0):(0,0,0),(1,0,1):(0,0,1),(1,0,0):(0,0,0),\
             (0,1,1):(0,0,1),(0,1,0):(0,0,0),(0,0,1):(0,0,0)}
        
        allPaths = dijkstra._get_paths(visitDict)
        correctPaths = \
            {(1,1,1):[(0,0,0),(0,0,1),(1,1,1)],(1,1,0):[(0,0,0),(1,1,0)]}
        self.assertDictEqual(allPaths,correctPaths)
                
        return None

class __call___(unittest.TestCase):#<--- yes, this class has 5 underscores in it
    def test_3d_grid_two_endpoints_default_return(self):
        def dist_func(coords,enegs,masses):
            return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
        x1 = np.array([0,1])
        x2 = np.array([0,2])
        x3 = np.array([1,3])
        
        coordMeshTuple = np.meshgrid(x1,x2,x3)
        zz = np.arange(8).reshape((2,2,2))
        initialPoint = np.array([0,0,1])
        finalPoint = np.array([[1,2,3],[1,2,1]])
        
        dijkstra = Dijkstra(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                            allowedEndpoints=finalPoint)
        
        pathIndsDict, pathArrDict, distDict = dijkstra(returnAll=True)
        
        correctPathInds = \
            {(1,2,3):[(0,0,0),(0,0,1),(1,1,1)],(1,2,1):[(0,0,0),(1,1,0)]}
        correctPaths = \
            {(1,2,3):np.array([[0.,0.,1.],[0.,0.,3.],[1.,2.,3.]]),\
             (1,2,1):np.array([[0.,0.,1.],[1.,2.,1.]])}
        #TODO: check that this is actually correct
        correctDistDict = {(1, 2, 3): 17.65247584249853, (1, 2, 1): 13.416407864998739}
        
        #Doing this to effectively compare the dictionaries, using SaneEqualityArray
        correctPathsForComparison = {}
        for key in correctPaths.keys():
            arr = correctPaths[key]
            saneArr = SaneEqualityArray(arr.shape,arr.dtype,arr)
            correctPathsForComparison[key] = saneArr
        
        self.assertDictEqual(correctPathInds,pathIndsDict)
        self.assertDictEqual(correctPathsForComparison,pathArrDict)
        warnings.warn("Still need to check that the distance dicts are equal")
                
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()