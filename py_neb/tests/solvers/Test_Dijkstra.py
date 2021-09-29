from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

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
    
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()