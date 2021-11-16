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

# class _select_next_point_(unittest.TestCase):
#     def test_2d_grid(self):
#         def dist_func(coords,enegs,masses):
#             return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
#         x1 = np.array([0.,1])
#         x2 = np.array([0.,0.5,1])
        
#         coordMeshTuple = np.meshgrid(x1,x2)
#         zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
#         initialPoint = np.array([0.,0])
#         finalPoint = np.array([1.,1])
        
#         dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
#                                 allowedEndpoints=finalPoint,logLevel=0)
            
#         currentInds = dp.initialInds
#         relativeNeighborInds = [(2,0),(1,0),(-1,0),(-2,0)]
        
#         minInds, minDist = dp._select_next_point(currentInds,relativeNeighborInds,0)
#         #Is the same setup as Test_Dijstra._construct_path_dict_.test_2d_grid
#         correctMinInds = (1,0)
#         correctMinDist = 0.5
        
#         self.assertEqual(minInds,correctMinInds)
#         self.assertEqual(minDist,correctMinDist)
        
#         return None
    
# class _call_truncated_(unittest.TestCase):
#     def test_truncated_grid(self):
#         #Note that DynamicProgramming uses fixed start and endpoints, and moves
#         #across the first coordinate in its search. So, given only two distinct
#         #first coordinates, it must draw a straight line from the starting point
#         #to the ending point.
#         def dist_func(coords,enegs,masses):
#             return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
#         x1 = np.array([0.,1])
#         x2 = np.array([0.,0.5,1])
        
#         coordMeshTuple = np.meshgrid(x1,x2)
#         zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
#         initialPoint = np.array([0.,0])
#         finalPoint = np.array([1.,1])
        
#         dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
#                                 allowedEndpoints=finalPoint,logLevel=0)
            
#         pathIndsDict, pathPtsDict, pathDistsDict = dp._call_truncated(np.array([3]))
        
#         correctPathIndsDict = {(1.,1.):SaneEqualityArray((2,2),dtype=int,buffer=np.array([[0,0],[2,1]]))}
#         correctPathPtsDict = {(1.,1.):SaneEqualityArray((2,2),buffer=np.array([[0.,0],[1.,1]]))}
#         correctPathDistsDict = {(1.,1.):3*np.sqrt(2)}
        
#         self.assertDictEqual(pathIndsDict,correctPathIndsDict)
#         self.assertDictEqual(pathPtsDict,correctPathPtsDict)
#         self.assertDictEqual(pathDistsDict,correctPathDistsDict)
        
#         return None
    
#     def test_larger_grid(self):
#         #Note that DynamicProgramming uses fixed start and endpoints, and moves
#         #across the first coordinate in its search. So, given only two distinct
#         #first coordinates, it must draw a straight line from the starting point
#         #to the ending point.
#         def dist_func(coords,enegs,masses):
#             return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
#         x1 = np.array([0.,0.3,0.6,1])
#         x2 = np.array([0.,0.5,1])
        
#         coordMeshTuple = np.meshgrid(x1,x2)
#         zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
#         initialPoint = np.array([0.,0])
#         finalPoint = np.array([1.,1])
        
#         dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
#                                 allowedEndpoints=finalPoint,logLevel=0)
            
#         pathIndsDict, pathPtsDict, pathDistsDict = dp._call_truncated(np.array([3]))
        
#         correctInds = np.array([[0,0],[0,1],[0,2],[2,3]])
#         correctPathIndsDict = {(1.,1.):SaneEqualityArray((4,2),dtype=int,\
#                                                          buffer=correctInds)}
#         correctPath = np.array([[0.,0],[0.3,0],[0.6,0],[1,1]])
#         correctPathPtsDict = {(1.,1.):SaneEqualityArray((4,2),buffer=correctPath)}
#         correctPathDistsDict = {(1.,1.):3.501098884280703} #Via Mathematica
        
#         self.assertDictEqual(pathIndsDict,correctPathIndsDict)
#         self.assertDictEqual(correctPathPtsDict,pathPtsDict)
#         self.assertDictEqual(pathDistsDict,correctPathDistsDict)
        
#         return None

class _gen_slice_inds_(unittest.TestCase):
    def test_standard(self):
        def dist_func(coords,enegs,masses):
            return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
        x1 = np.array([0.,0.3,0.6,1])
        x2 = np.array([0.,0.5,1])
        
        coordMeshTuple = np.meshgrid(x1,x2)
        zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
        initialPoint = np.array([0.,0])
        finalPoint = np.array([1.,1])
        
        dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                                allowedEndpoints=finalPoint,logLevel=0)
            
        sliceInds = dp._gen_slice_inds(3)
        correctInds = [(0,3),(1,3),(2,3)]
        
        self.assertEqual(sliceInds, correctInds)
        
        return None
    
class _select_prior_points_(unittest.TestCase):
    def test_larger_grid(self):
        #Note that DynamicProgramming uses fixed start and endpoints, and moves
        #across the first coordinate in its search. So, given only two distinct
        #first coordinates, it must draw a straight line from the starting point
        #to the ending point.
        def dist_func(coords,enegs,masses):
            return enegs[1]*np.linalg.norm(coords[1]-coords[0]), enegs, masses
        
        x1 = np.array([0.,0.3,0.6,1])
        x2 = np.array([0.,0.5,1])
        
        coordMeshTuple = np.meshgrid(x1,x2)
        zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
        initialPoint = np.array([0.,0])
        finalPoint = np.array([1.,1])
        
        dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                                allowedEndpoints=finalPoint,logLevel=0)
            
        previousIndsArr = -1*np.ones(zz.shape+(2,),dtype=int)
        distArr = np.inf*np.ones(zz.shape)
        currentIdx = 1
        
        newIndsArr, newDistArr = dp._select_prior_points(currentIdx,previousIndsArr,distArr)
        
        correctNewIndsArr = previousIndsArr.copy()
        correctNewIndsArr[0,1] = [0,0]
        correctNewIndsArr[1,1] = [1,0]
        correctNewIndsArr[2,1] = [2,0]
        
        self.assertIsNone(np.testing.assert_array_equal(newIndsArr,correctNewIndsArr))
        
        correctNewDistArr = distArr.copy()
        correctNewDistArr[0,1] = 0.09
        correctNewDistArr[1,1] = 0.39
        correctNewDistArr[2,1] = 0.69
        
        self.assertIsNone(np.testing.assert_array_equal(newDistArr,correctNewDistArr))
        
        return None
    
class __call___(unittest.TestCase):
    def test_larger_grid(self):
        def dist_func(coords,enegs,masses):
            val = 0
            for ptIter in range(1,coords.shape[0]):
                val += np.sqrt(enegs[ptIter])*np.linalg.norm(coords[ptIter]-coords[ptIter-1])
            
            return val, enegs, masses
        
        x1 = np.array([0.,0.3,0.6,1])
        x2 = np.array([0.,0.5,1])
        
        coordMeshTuple = np.meshgrid(x1,x2)
        zz = coordMeshTuple[0] + 2*coordMeshTuple[1] #Is x+2y
        initialPoint = np.array([0.,0])
        finalPoint = np.array([1.,1])
        
        dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                                allowedEndpoints=finalPoint,logLevel=0)
            
        minIndsDict, minPathDict, distsDict = dp()
        
        allPaths = np.array([[[0,0]]+[[0.3,x2[i]]]+[[0.6,x2[j]]]+[[1,1]] for i in range(3) \
                             for j in range(3)])
        acts = [dist_func(p,p[:,0]+2*p[:,1],None)[0] for p in allPaths]
        print(acts)
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()