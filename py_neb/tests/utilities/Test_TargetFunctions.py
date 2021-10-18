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
class action_(unittest.TestCase):
    def test_constant_mass_array_potential(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        
        act, eneg, mass = TargetFunctions.action(path,potential)
        
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
        
        act, eneg, mass = TargetFunctions.action(path,pot)
        
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
        
        act, eneg, mass = TargetFunctions.action(path,potential,masses=mass)
        
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
        
        act, eneg, mass = TargetFunctions.action(path,pot,masses=mass)
        
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
        
        act, eneg, mass = TargetFunctions.action(path,potential,masses=mass_func)
        
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
        
        act, eneg, mass = TargetFunctions.action(path,pot,masses=mass_func)
        
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
            TargetFunctions.action(path,potential)
        
        return None
    
    def test_wrong_mass_shape(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        massGrid = np.ones(12)
        
        with self.assertRaises(ValueError):
            TargetFunctions.action(path,potential,masses=massGrid)
        
        return None
    
class term_in_action_sum_(unittest.TestCase):
    def test_array_potential_none_mass(self):
        path = np.arange(4).reshape((2,2))
        potential = np.array([2])
        
        act, eneg, mass = TargetFunctions.term_in_action_sum(path,potential)
        
        correctAction = 4*np.sqrt(2)
        correctMass = np.identity(2)
        
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_scalar_potential_function_mass(self):
        def mass(point):
            return np.array([[1.,1.],[0.,2.]])
        
        path = np.arange(4).reshape((2,2))
        potential = 2
        
        act, eneg, mass = TargetFunctions.term_in_action_sum(path,potential,\
                                                              masses=mass)
        
        correctAction = 8.
        correctMass = np.array([[1.,1.],[0.,2.]])
        
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_function_potential_array_mass(self):
        def pot(point):
            return np.array([2])
        
        path = np.arange(4).reshape((2,2))
        mass = np.array([[1.,1.],[0.,2.]])
        
        act, eneg, mass = TargetFunctions.term_in_action_sum(path,pot,\
                                                             masses=mass)
        
        correctAction = 8.
        correctMass = np.array([[1.,1.],[0.,2.]])
        
        self.assertEqual(act,correctAction)
        self.assertIsNone(np.testing.assert_array_equal(eneg,np.array([2])))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_wrong_number_of_points(self):
        points = np.arange(12).reshape((6,2))
        potential = np.arange(6)
        
        with self.assertRaises(ValueError):
            TargetFunctions.term_in_action_sum(points,potential)
            
        return None
    
    def test_wrong_pot_arr_shape(self):
        points = np.arange(4).reshape((2,2))
        potential = np.arange(2)
        
        with self.assertRaises(ValueError):
            TargetFunctions.term_in_action_sum(points,potential)
            
        return None
    
    def test_wrong_mass_arr_shape(self):
        points = np.arange(4).reshape((2,2))
        potential = 2
        mass = np.full((3,2,2),np.identity(2))
        
        with self.assertRaises(ValueError):
            TargetFunctions.term_in_action_sum(points,potential,masses=mass)
            
        return None

class action_squared_(unittest.TestCase):
    def test_constant_mass_array_potential(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        
        actSqr, eneg, mass = TargetFunctions.action_squared(path,potential)
        
        correctActionSqr = 240
        correctMass = np.full((5,2,2),np.identity(2))
        
        self.assertEqual(actSqr,correctActionSqr)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_constant_mass_function_potential(self):
        path = np.arange(6).reshape((3,2))
        
        def pot(coordsArr):
            return np.ones(3)
        
        actSqr, eneg, mass = TargetFunctions.action_squared(path,pot)
        
        correctActionSqr = 16
        correctPotential = np.ones(3)
        correctMass = np.full((3,2,2),np.identity(2))
        
        #Also checked this by hand
        self.assertEqual(actSqr,correctActionSqr)
        self.assertIsNone(np.testing.assert_array_equal(eneg,correctPotential))
        self.assertIsNone(np.testing.assert_array_equal(mass,correctMass))
        
        return None
    
    def test_grid_mass_array_potential(self):
        path = np.arange(6).reshape((3,2))
        potential = np.arange(3)**2
        mass = np.full((2,2,2),np.identity(2))
        mass = np.vstack((mass,np.array([[[1.,1.],\
                                          [0.,2.]]])))
        
        actSqr, eneg, massOut = TargetFunctions.action_squared(path,potential,masses=mass)
        
        correctActionSqr = 72
        
        self.assertEqual(actSqr,correctActionSqr)
        self.assertIsNone(np.testing.assert_array_equal(eneg,potential))
        self.assertIsNone(np.testing.assert_array_equal(mass,massOut))
        
        return None
    
    def test_grid_mass_function_potential(self):
        path = np.arange(6).reshape((3,2))
        mass = np.full((3,2,2),np.identity(2))
        
        def pot(coordsArr):
            return np.ones(3)
        
        act, eneg, mass = TargetFunctions.action_squared(path,pot,masses=mass)
        
        correctAction = 16
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
        
        act, eneg, mass = TargetFunctions.action_squared(path,potential,masses=mass_func)
        
        correctAction = 240
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
        
        act, eneg, mass = TargetFunctions.action_squared(path,pot,masses=mass_func)
        
        correctAction = 16
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
            TargetFunctions.action_squared(path,potential)
        
        return None
    
    def test_wrong_mass_shape(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(5)**2
        massGrid = np.ones(12)
        
        with self.assertRaises(ValueError):
            TargetFunctions.action_squared(path,potential,masses=massGrid)
        
        return None
    
class mep_default_(unittest.TestCase):
    def test_potential_function(self):
        path = np.arange(4).reshape((2,2))
        def potential(path):
            return path[:,0]**2 + path[:,1]**2
        def auxFunc(path):
            return path[:,0]**3 + path[:,1]**3
        
        eneg, aux_eneg = TargetFunctions.mep_default(path,potential,auxFunc=auxFunc)
        
        correctEneg = np.array([1,13])
        correctAux = np.array([1,35])
        
        self.assertIsNone(np.testing.assert_array_equal(eneg,correctEneg))
        self.assertIsNone(np.testing.assert_array_equal(aux_eneg,correctAux))
        
        return None
    
    def test_wrong_potential_shape(self):
        path = np.arange(10).reshape((5,2))
        potential = np.arange(30)**2
        auxFunc = np.arange(30)**3
        
        with self.assertRaises(ValueError):
            TargetFunctions.mep_default(path,potential,auxFunc)
            
        return None
    
if __name__ == "__main__":
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore",message=".*should_run_async.*")
    unittest.main()