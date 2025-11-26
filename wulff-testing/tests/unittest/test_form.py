#!/usr/bin/env python

import unittest

import numpy as np
from wulffpack.core import Form
from wulffpack.core.form import setup_forms


class TestForm(unittest.TestCase):
    """Test of class Form"""

    def __init__(self, *args, **kwargs):
        super(TestForm, self).__init__(*args, **kwargs)
        self.miller_indices = (1, 1, -1)
        self.energy = 3.14
        self.parent_miller_indices = (1, 1, 1)
        self.reciprocal_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.symmetries = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                           np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                           np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])]

        self.vertices = []
        self.vertices.append(np.array([3.0, -1.2, -1.2]))
        self.vertices.append(np.array([3.0, 1.2, 1.2]))
        self.vertices.append(np.array([0.0, 4.7, 1.7]))

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.form = Form(self.miller_indices, self.energy,
                         self.reciprocal_cell, self.symmetries,
                         self.parent_miller_indices)

    def test_init(self):
        """Tests that initialization of tested class works."""
        form = Form(self.miller_indices, self.energy,
                    self.reciprocal_cell, self.symmetries,
                    self.parent_miller_indices)
        self.assertIsInstance(form, Form)
        self.assertEqual(form.miller_indices, self.miller_indices)
        self.assertEqual(form.parent_miller_indices,
                         self.parent_miller_indices)
        self.assertEqual(form.energy, self.energy)

        # Test that the right facets were created
        self.assertEqual(len(form.facets), 2)
        target_normal = np.array([0.57735027, 0.57735027, -0.57735027])
        self.assertTrue(np.allclose(form.facets[0].normal, target_normal))
        target_normal *= -1
        self.assertTrue(np.allclose(form.facets[1].normal, target_normal))

        # Test that the right symmetries were saved in those facets
        symmetries = self.form.facets[0].symmetries
        self.assertEqual(len(symmetries), 1)
        self.assertTrue(np.allclose(symmetries[0], self.symmetries[0]))

        symmetries = self.form.facets[1].symmetries
        self.assertEqual(len(symmetries), 2)
        self.assertTrue(np.allclose(symmetries[0], self.symmetries[1]))
        self.assertTrue(np.allclose(symmetries[1], self.symmetries[2]))

    def test_area(self):
        """Tests area property"""
        for vertex in self.vertices:
            self.form.facets[0].add_vertex(vertex)

        # Target area (divide by two but then multiply by two because two
        # facets)
        target = np.linalg.norm(np.cross(self.vertices[2] - self.vertices[0],
                                         self.vertices[1] - self.vertices[0]))

        self.assertAlmostEqual(self.form.area, target)

    def test_surface_energy(self):
        """Tests surface energy property"""
        for vertex in self.vertices:
            self.form.facets[0].add_vertex(vertex)

        # Target area (divide by two but then multiply by two because two
        # facets)
        target = np.linalg.norm(np.cross(self.vertices[2] - self.vertices[0],
                                         self.vertices[1] - self.vertices[0]))
        target *= self.energy

        self.assertAlmostEqual(self.form.surface_energy, target)

    def test_volume(self):
        """Tests volume property"""
        for vertex in self.vertices:
            self.form.facets[0].add_vertex(vertex)

        # Target area (divide by two but then multiply by two because two
        # facets)
        area = np.linalg.norm(np.cross(self.vertices[2] - self.vertices[0],
                                       self.vertices[1] - self.vertices[0]))
        normal = np.array([0.57735027, 0.57735027, -0.57735027])
        height = np.dot(self.vertices[0], normal)

        self.assertAlmostEqual(self.form.volume, area * height / 3)

    def test_setup_forms(self):
        """Tests setup forms function"""
        surface_energies = {(1, 1, 1): 1.2, (1, 0, 0): 1.3,
                            'twin': 0.1, 'interface': 0.3}
        cell = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        symmetries_restricted = [np.eye(3), -np.eye(3)]
        symmetries_full = [np.eye(3), -np.eye(3),
                           np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]
        twin_boundary = (-1, 1, 1)
        interface = (3, 1, 1)
        forms = setup_forms(surface_energies,
                            cell=cell,
                            symmetries_restricted=symmetries_restricted,
                            symmetries_full=symmetries_full)
        self.assertEqual(len(forms), 4)
        self.assertIsInstance(forms[0], Form)
        for form in forms:
            self.assertEqual(len(form.facets), 2)

        forms = setup_forms(surface_energies,
                            cell=cell,
                            symmetries_restricted=symmetries_restricted,
                            symmetries_full=symmetries_full,
                            twin_boundary=twin_boundary)
        self.assertEqual(len(forms), 5)
        self.assertIsInstance(forms[0], Form)
        for form in forms:
            self.assertEqual(len(form.facets), 2)

        forms = setup_forms(surface_energies,
                            cell=cell,
                            symmetries_restricted=symmetries_restricted,
                            symmetries_full=symmetries_full,
                            interface=interface)
        self.assertEqual(len(forms), 5)
        self.assertIsInstance(forms[0], Form)
        for form in forms:
            self.assertEqual(len(form.facets), 2)

        # Test that it raises exception when using negative energies
        with self.assertRaises(ValueError) as cm:
            setup_forms({(1, 1, 1): 1.2, 'twin': -0.5},
                        cell=cell,
                        symmetries_restricted=symmetries_restricted,
                        symmetries_full=symmetries_full,
                        interface=interface)
        self.assertIn('Please use only positive', str(cm.exception))

        # Test that exception is raised when using
        # symmetrically equivalent facets
        surface_energies[(-1, 0, 0)] = 2.0
        with self.assertRaises(ValueError) as cm:
            setup_forms(surface_energies,
                        cell=cell,
                        symmetries_restricted=symmetries_restricted,
                        symmetries_full=symmetries_full)
        self.assertIn('(-1, 0, 0) are equivalent to (1, 0, 0) by symmetry', str(cm.exception))

        # The same thing goes if restricted and full symmetries are the same
        with self.assertRaises(ValueError) as cm:
            setup_forms(surface_energies,
                        cell=cell,
                        symmetries_restricted=symmetries_full,
                        symmetries_full=symmetries_full)
        self.assertIn('(-1, 0, 0) are equivalent to (1, 0, 0) by symmetry', str(cm.exception))

        # It should work if energies are the same
        surface_energies[(-1, 0, 0)] = surface_energies[(1, 0, 0)]
        forms = setup_forms(surface_energies,
                            cell=cell,
                            symmetries_restricted=symmetries_restricted,
                            symmetries_full=symmetries_full)
        self.assertEqual(len(forms), 4)
        self.assertIsInstance(forms[0], Form)
        for form in forms:
            self.assertEqual(len(form.facets), 2)

    def test_edge_length(self):
        """Tests edge length property"""
        for vertex in self.vertices:
            self.form.facets[0].add_vertex(vertex)
            self.form.facets[1].add_vertex(vertex)
        target_value = 2 * (np.linalg.norm(self.vertices[1] - self.vertices[0]) +
                            np.linalg.norm(self.vertices[2] - self.vertices[0]) +
                            np.linalg.norm(self.vertices[2] - self.vertices[1]))
        self.assertEqual(self.form.edge_length, target_value)


if __name__ == '__main__':
    unittest.main()
