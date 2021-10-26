from context import *

import unittest
import warnings

#pyNebDir imported from context; perhaps bad practice?
print("\nRunning "+os.path.relpath(__file__,pyNebDir))

class __init___(unittest.TestCase):
    #Don't know how exactly to automatically test this... don't want
    #to construct splines by hand. Maybe just rely (here) on splprep
    #having its own tests
    def test_1d_curve(self):
        x = np.linspace(0,1,10)
        path = np.sin(4*np.pi*x)
        interpPath = InterpolatedPath(path,kwargs={"s":0,"u":x})
        
        t = np.linspace(0,1,100)
        pts = interpPath(t)
        
        fig, ax = plt.subplots()
        ax.plot(x,path,"bo")
        ax.plot(t,pts[0],"r.-")
        
        doublePts = np.arange(0,1+1/18,1/18)
        doublePath = interpPath(doublePts)
        
        ax.plot(doublePts,doublePath[0],"g-")
        
        return None
    
    def test_2d_curve(self):
        #Example taken from scipy.interpolate.splprep docs
        phi = np.linspace(0,2*np.pi,40)
        r = 0.5 + np.cos(phi)
        x, y = r*np.cos(phi), r*np.sin(phi)
        path = np.array([x,y]).T
        
        interpPath = InterpolatedPath(path,kwargs={"s":0,"u":phi/(2*np.pi)})
        
        fig, ax = plt.subplots()
        ax.plot(*path.T,"ro")
        
        t = np.linspace(0,1,200)
        newPoints = interpPath(t)
        
        ax.plot(*newPoints,"b.")
        return None
    
class compute_along_path_(unittest.TestCase):
    def test_1d_arc_length(self):
        def target_func(curve):
            #Arc length of 1D curve (not embedded in 2D)
            out = 0.
            for i in range(1,len(curve)):
                out += abs(curve[i] - curve[i-1])
            return out
        
        t = np.linspace(0,1,20)
        path = t**0.2
        interpPath = InterpolatedPath(path,kwargs={"s":0,"u":t})
        
        pathOut, tfOut = interpPath.compute_along_path(target_func,300)
        
        correctLength = np.array([1.])
        self.assertIsNone(np.testing.assert_array_equal(correctLength,tfOut))
        
        #Not sure how to automate that the path is doing as expected, but
        #this plot will show whether or not it is
        fig, ax = plt.subplots()
        ax.plot(t,path,"b.")
        
        tDense = np.linspace(0,1,300)
        ax.plot(tDense,pathOut.flatten(),"r-")
        
        ax.plot(tDense,tDense**0.2,"g-")
        
        return None
    
if __name__ == "__main__":
    # warnings.simplefilter("ignore")
    unittest.main()