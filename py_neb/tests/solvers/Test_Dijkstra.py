from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class _construct_path_dict_(unittest.TestCase):
    def test_3d_grid(self):
        def dist_func(coords):
            return np.linalg.norm(coords[1]-coords[0])
        
        x1 = np.array([0,1])
        x2 = np.array([0,2])
        x3 = np.array([1,3])
        
        coordMeshTuple = np.meshgrid(x1,x2,x3)
        zz = np.arange(8).reshape((2,2,2))
        initialPoint = np.array([0,0,1])
        finalPoint = np.array([1,2,3])
        
        dijkstra = Dijkstra(initialPoint,coordMeshTuple,zz,target_func=dist_func,\
                            allowedEndpoints=finalPoint)
        # dijkstra._
        
        return None
    
    def test_2d_gauss_with_poly(self):
        #Don't need anything more complicated than a function taking in a meshgrid
        def dummy_func(meshGrid):
            x, y = meshGrid
            return x*(1-2*np.exp(-((x-2)**2+y**2)/0.2)) + 1.9
                
        x = np.arange(-5,2.5,0.5)
        y = np.arange(-1,1.5,0.5)
        
        coordMeshTuple = np.meshgrid(x,y)
        zz = dummy_func(coordMeshTuple)
        # print(zz.size)
        initialPoint = np.array([2.,0.])
        
        dijkstra = Dijkstra(initialPoint,coordMeshTuple,zz)
        
        tentativeDistance, neighborsVisitDict, \
            endpointIndsList = dijkstra._construct_path_dict()
        
        # print(tentativeDistance)
        
        fig, ax = plt.subplots()
        ax.contourf(*coordMeshTuple,tentativeDistance.data)
        
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()