#!/usr/bin/env python

import unittest

import numpy as np
from ase.build import bulk
from wulffpack.core.geometry import (get_tetrahedral_volume,
                                     get_angle,
                                     get_rotation_matrix,
                                     break_symmetry,
                                     get_symmetries,
                                     is_array_in_arrays,
                                     where_is_array_in_arrays)


class TestGeometry(unittest.TestCase):
    """Test of geometry module"""

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_get_tetrahedral_volume(self):
        """Tests calculation of volume from four vertices"""
        triangle = [np.array([3., 0., 0.]),
                    np.array([0., 3., 0.]),
                    np.array([0., 0., 3.])]
        origin = np.array([0., 0., 0.])
        volume = get_tetrahedral_volume(triangle, origin=origin)
        self.assertAlmostEqual(volume, 3**3 / 6)

        with self.assertRaises(ValueError) as cm:
            get_tetrahedral_volume(triangle[:2], origin=origin)
        self.assertIn('must contain three coordinates', str(cm.exception))

    def test_get_angle(self):
        """Tests angle calculation functionality"""
        v1 = (1., 0., 0.)
        v2 = (1., 0., 0.)
        self.assertAlmostEqual(get_angle(v1, v2), 0.0)

        v1 = (-1., 0., 0.)
        v2 = (1., 0., 0.)
        self.assertAlmostEqual(get_angle(v1, v2), np.pi)

    def test_get_rotation_matrix(self):
        """Tests calculation of rotation matrix"""
        theta = np.pi
        u = np.array([0., 0., 1.])
        target = np.array([[-1., 0., 0.],
                           [0., -1., 0.],
                           [0., 0., 1.]])
        retval = get_rotation_matrix(theta, u)
        self.assertTrue(np.allclose(retval, target))

        theta = np.pi / 3
        u = np.array([1., 0., 0.])
        target = np.array([[1., 0., 0.],
                           [0., 0.5, - 3**0.5 / 2],
                           [0., 3**0.5 / 2, 0.5]])

        retval = get_rotation_matrix(theta, u)
        self.assertTrue(np.allclose(retval, target))

    def test_break_symmetry(self):
        """Tests function that breaks symmetry"""
        full_symmetry = [np.eye(3), -np.eye(3),
                         np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])]
        symmetry_axes = [np.array([1., 0., 0.])]
        symmetries = break_symmetry(full_symmetry, symmetry_axes)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 1)
        self.assertIsInstance(symmetries[0], np.ndarray)

        symmetry_axes = [np.array([1., 1., 1.])]
        symmetries = break_symmetry(full_symmetry, symmetry_axes)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 1)
        self.assertIsInstance(symmetries[0], np.ndarray)

        symmetries = break_symmetry(full_symmetry, symmetry_axes, inversion=[True])
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 2)
        self.assertIsInstance(symmetries[0], np.ndarray)

        symmetry_axes = [np.array([1., 1., 1.]), np.array([0., 1., 1.])]
        symmetries = break_symmetry(full_symmetry, symmetry_axes, inversion=[True, False])
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 1)
        self.assertIsInstance(symmetries[0], np.ndarray)

        with self.assertRaises(ValueError) as cm:
            break_symmetry(full_symmetry, symmetry_axes, inversion=[True])
        self.assertIn('must be a list of bools', str(cm.exception))

    def test_get_symmetries(self):
        """Tests get primitive with symmetries function"""
        # FCC, cubic cell
        prim = bulk('Ni', a=3.9, crystalstructure='fcc', cubic=True)
        symmetries = get_symmetries(prim)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 48)
        self.assertIsInstance(symmetries[0], np.ndarray)

        # FCC, primitive cell
        prim = bulk('Ag', a=4.2, crystalstructure='fcc')
        symmetries = get_symmetries(prim)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 48)
        self.assertIsInstance(symmetries[0], np.ndarray)

        # BCC
        prim = bulk('Au', a=3.0, crystalstructure='bcc')
        symmetries = get_symmetries(prim)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 48)
        self.assertIsInstance(symmetries[0], np.ndarray)

        # SC
        prim = bulk('Au', a=3.0, crystalstructure='sc')
        symmetries = get_symmetries(prim)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 48)
        self.assertIsInstance(symmetries[0], np.ndarray)

        # Something else
        prim = bulk('Au', a=4.0, crystalstructure='sc')
        prim.set_cell(np.array([[1.2, 0.1, 0.], [0., 1., 0.], [0., 0., 1.]]))
        symmetries = get_symmetries(prim)
        self.assertIsInstance(symmetries, list)
        self.assertEqual(len(symmetries), 4)
        self.assertIsInstance(symmetries[0], np.ndarray)

    def test_is_array_in_arrays(self):
        """Tests check whether array is in list of arrays"""
        arrays = [np.array([0.0, 1.2, 1.3]),
                  np.array([0.1, 1.1, 1.5])]
        self.assertTrue(is_array_in_arrays(arrays[0], arrays))
        self.assertFalse(is_array_in_arrays(np.array([0.0, 0.1, 0.1]), arrays))

    def test_where_is_array_in_array(self):
        """Tests search for array in list of arrays"""
        arrays = [np.array([0.2, 0.1, 0.2]),
                  np.array([0.3, 0.0, -1.0]),
                  np.array([0.0, 0.1, 0.2])]
        self.assertEqual(where_is_array_in_arrays(np.array([0.3, 0.0, -1.0]), arrays), 1)
        self.assertEqual(where_is_array_in_arrays(np.array([0.5, 0.5, -0.5]), arrays), -1)
