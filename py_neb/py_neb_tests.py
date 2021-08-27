from py_neb import *
import unittest

"""
Use module unittest (see https://docs.python.org/3/library/unittest.html#module-unittest)
    for unit tests. It's much simpler.
    
    
List of tests to add:
    -Test find_local_minimum, using a 3D polynomial with known minima
    -Test midpoint_grad with polynomial:
        *With a single point, to test reshaping
        *With many points, just to check for sure
    -Action:
        *Function mass/potential with wrong size outputs?
        
Tests added:
    -Action:
        *Constant mass, function potential
        *Grid mass, array potential
        *Grid mass, function potential
        *Function mass, array potential
        *Function mass, function potential
        *Array mass of wrong size for path, to check one error
        *Array potential of wrong size for path, to check other error
"""

# class find_local_minimum_:
#     @staticmethod
#     def standard_arr():
#         return False

class action_(unittest.TestCase):
    def test_constant_mass_array_potential(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        
        act, eneg, mass = action(path,potential)
        
        correctAction = 40
        correctMass = np.full((5,2,2),np.identity(2))
        
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_constant_mass_function_potential(self):
        path = np.arange(6).reshape((3,2))
        
        def pot(coordsArr):
            return np.ones(3)
        
        act, eneg, mass = action(path,pot)
        
        correctAction = 8
        correctPotential = np.ones(3)
        correctMass = np.full((3,2,2),np.identity(2))
        
        #Also checked this by hand
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,correctPotential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_grid_mass_array_potential(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        mass = np.full((5,2,2),np.identity(2))
        
        act, eneg, mass = action(path,potential)
        
        correctAction = 40
        correctMass = np.full((5,2,2),np.identity(2))
        
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_grid_mass_function_potential(self):
        path = np.arange(6).reshape((3,2))
        mass = np.full((3,2,2),np.identity(2))
        
        def pot(coordsArr):
            return np.ones(3)
        
        act, eneg, mass = action(path,pot,masses=mass)
        
        correctAction = 8
        correctPotential = np.ones(3)
        correctMass = np.full((3,2,2),np.identity(2))
        
        #Also checked this by hand
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,correctPotential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_function_mass_array_potential(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        
        def mass_func(coordsArr):
            return np.full((5,2,2),np.identity(2))
        
        act, eneg, mass = action(path,potential,masses=mass_func)
        
        correctAction = 40
        correctMass = np.full((5,2,2),np.identity(2))
        
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_function_mass_function_potential(self):
        path = np.arange(6).reshape((3,2))
        
        def pot(coordsArr):
            return np.ones(3)
        
        def mass_func(coordsArr):
            return np.full((3,2,2),np.identity(2))
        
        act, eneg, mass = action(path,pot,masses=mass_func)
        
        correctAction = 8
        correctPotential = np.ones(3)
        correctMass = np.full((3,2,2),np.identity(2))
        
        #Also checked this by hand
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,correctPotential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_wrong_potential_shape(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(30)**2
        
        with self.assertRaises(ValueError):
            action(path,potential)
        
        return None
    
    def test_wrong_mass_shape(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        massGrid = np.ones(12)
        
        with self.assertRaises(ValueError):
            action(path,potential,masses=massGrid)
        
        return None
    # @staticmethod
    # def constant_mass_function_potential():
    #     return False
    
# class Utilities_Validation:
#     @staticmethod
#     def val_find_local_minimum():
#         dsets, attrs = \
#             FileIO.read_from_h5("Test_PES.h5","Daniels_Code/Test_Files/")
                
#         minInds = Utilities.find_local_minimum(dsets["PES"])
#         fig, ax = Utilities.standard_pes(dsets["Q20"],dsets["Q30"],dsets["PES"])
        
#         ax.scatter(dsets["Q20"][minInds],dsets["Q30"][minInds],color="k",marker="x")
#         ax.set(xlim=(dsets["Q20"].min(),dsets["Q20"].max()),\
#                ylim=(dsets["Q30"].min(),dsets["Q30"].max()))
            
#         testFolder = "Test_Outputs/Utilities/"
#         if not os.path.isdir(testFolder):
#             os.makedirs(testFolder)
            
#         fig.savefig(testFolder+"val_find_local_minimum.pdf")
        
#         return None
    
if __name__ == "__main__":
    unittest.main()
