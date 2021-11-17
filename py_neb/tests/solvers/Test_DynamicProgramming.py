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
        distArr[dp.initialInds] = 0
        currentIdx = 1
        
        newIndsArr, newDistArr = dp._select_prior_points(currentIdx,previousIndsArr,distArr)
        # print(newIndsArr)
        # print(newDistArr)
        correctNewIndsArr = previousIndsArr.copy()
        correctNewIndsArr[0,1] = [0,0]
        correctNewIndsArr[1,1] = [0,0]
        correctNewIndsArr[2,1] = [0,0]
        
        self.assertIsNone(np.testing.assert_array_equal(newIndsArr,correctNewIndsArr))
        
        #Verified in Mathematica
        correctNewDistArr = distArr.copy()
        correctNewDistArr[0,1] = 0.09
        correctNewDistArr[1,1] = 0.758023746329889
        correctNewDistArr[2,1] = 2.401270497049426
        
        self.assertIsNone(np.testing.assert_allclose(newDistArr,correctNewDistArr))
        
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
        
        correctMinIndsDict = {tuple(finalPoint):[(0,0),(0,1),(1,2),(2,3)]}
        self.assertEqual(minIndsDict,correctMinIndsDict)
        
        correctMinPath = np.array([[0,0],[0.3,0],[0.6,0.5],[1,1]])
        correctMinPathDict = {tuple(finalPoint):SaneEqualityArray((4,2),buffer=correctMinPath)}
        self.assertEqual(minPathDict,correctMinPathDict)
        
        #Verified in Mathematica to floating point precision
        correctDistDict = {tuple(finalPoint):2.0109339744759227}
        self.assertEqual(distsDict,correctDistDict)
        
        return None
    
    def test_two_endpoints(self):
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
        finalPoints = np.array([[1.,1],[0.6,1]])
        
        dp = DynamicProgramming(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                                allowedEndpoints=finalPoints,logLevel=0)
            
        minIndsDict, minPathDict, distsDict = dp()
        
        correctMinIndsDict = {tuple(finalPoints[0]):[(0,0),(0,1),(1,2),(2,3)],\
                              tuple(finalPoints[1]):[(0,0),(1,1),(2,2)]}
        self.assertEqual(minIndsDict,correctMinIndsDict)
        
        p1 = np.array([[0,0],[0.3,0],[0.6,0.5],[1,1]])
        p2 = np.array([[0,0],[0.3,0.5],[0.6,1]])
        correctMinPathDict = {tuple(finalPoints[0]):SaneEqualityArray((4,2),buffer=p1),\
                              tuple(finalPoints[1]):SaneEqualityArray((3,2),buffer=p2)}
        self.assertEqual(minPathDict,correctMinPathDict)
        
        #Verified in Mathematica to floating point precision
        correctDistDict = {tuple(finalPoints[0]):2.0109339744759227,\
                            tuple(finalPoints[1]):1.6050435474272389}
        self.assertEqual(distsDict,correctDistDict)
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()