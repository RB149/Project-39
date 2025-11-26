#!/usr/bin/env python

import unittest

import numpy as np
from wulffpack.core import Facet


class TestFacet(unittest.TestCase):
    """Test of base class Facet"""

    def __init__(self, *args, **kwargs):
        super(TestFacet, self).__init__(*args, **kwargs)
        self.normal = (3, 2, 1)
        self.energy = 3.14
        self.symmetry = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.original_grain = True

        self.vertices = []
        self.vertices.append(np.array([3.0, 1.2, -1.0]))
        self.vertices.append(np.array([3.0, 0.2, 1.0]))
        self.vertices.append(np.array([0.0, 4.7, 1.0]))
        self.vertices.append((self.vertices[0] + self.vertices[1]) / 2)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.facet = Facet(self.normal, self.energy,
                           self.symmetry, self.original_grain)

    def test_init(self):
        """Tests that initialization of tested class works."""
        facet = Facet(self.normal, self.energy,
                      self.symmetry, self.original_grain)
        self.assertIsInstance(facet, Facet)
        self.assertTrue(np.allclose(
            facet.normal, [0.80178373, 0.53452248, 0.26726124]))
        self.assertEqual(len(facet.vertices), 0)
        self.assertEqual(len(facet.original_vertices), 0)
        self.assertEqual(len(facet.symmetries), 1)
        self.assertTrue(np.allclose(facet.symmetries[0], self.symmetry))

    def test_add_vertex(self):
        """
        Tests that vertices can be added but only if they are unique and lie
        in the same plane as previously added vertices.
        """
        self.facet.add_vertex(self.vertices[0])
        self.assertEqual(len(self.facet.vertices), 1)
        self.assertTrue(np.allclose(self.facet.vertices[0], self.vertices[0]))
        self.facet.add_vertex(self.vertices[0])
        self.assertEqual(len(self.facet.vertices), 1)
        self.facet.add_vertex(self.vertices[1])
        self.assertEqual(len(self.facet.vertices), 2)
        self.assertTrue(np.allclose(self.facet.vertices[0], self.vertices[0]))
        self.assertTrue(np.allclose(self.facet.vertices[1], self.vertices[1]))
        self.facet.add_vertex(self.vertices[0])
        self.facet.add_vertex(self.vertices[1])
        self.assertEqual(len(self.facet.vertices), 2)

        with self.assertRaises(ValueError) as cm:
            self.facet.add_vertex([1.2, 1.2, 1.2])
        self.assertIn('does not lie in the same plane', str(cm.exception))

    def test_distance_from_origin(self):
        """Tests distance to facet from origin property"""
        for vertex in self.vertices:
            self.facet.add_vertex(vertex)
        self.assertAlmostEqual(self.facet.distance_from_origin, 2.7795169159)

    def test_remove_redundant_vertices(self):
        """Tests removal of vertices that lie midway between two other vertices"""
        for vertex in self.vertices:
            self.facet.add_vertex(vertex)
        self.assertEqual(len(self.facet.vertices), 4)
        self.facet.remove_redundant_vertices()
        self.assertEqual(len(self.facet.vertices), 3)
        self.facet.add_vertex(self.vertices[3])
        self.assertEqual(len(self.facet.vertices), 4)

    def test_ordered_vertices(self):
        """Tests ordered vertices property"""
        for vertex in self.vertices[:3]:
            self.facet.add_vertex(vertex)
        ordered_vertices = self.facet.ordered_vertices
        self.assertEqual(len(ordered_vertices), len(self.vertices))
        self.assertTrue(np.allclose(ordered_vertices[0], ordered_vertices[-1]))

    def test_face_as_triangles(self):
        """Tests face as triangles propert"""
        for vertex in self.vertices[:3]:
            self.facet.add_vertex(vertex)
        triangles = self.facet.face_as_triangles
        self.assertEqual(len(triangles), len(self.vertices) - 1)
        for triangle in triangles:
            self.assertEqual(len(triangle), 3)

    def test_area(self):
        """Tests area property"""
        for vertex in self.vertices[:3]:
            self.facet.add_vertex(vertex)
        target = np.linalg.norm(np.cross(self.vertices[2] - self.vertices[0],
                                         self.vertices[1] - self.vertices[0])) / 2
        self.assertAlmostEqual(self.facet.area, target)

    def test_surface_energy(self):
        """Tests surface energy property"""
        for vertex in self.vertices[:3]:
            self.facet.add_vertex(vertex)
        target = np.linalg.norm(np.cross(self.vertices[2] - self.vertices[0],
                                         self.vertices[1] - self.vertices[0])) / 2
        target *= self.energy
        self.assertAlmostEqual(self.facet.surface_energy, target)

    def test_get_volume(self):
        """Tests get volume function"""
        for vertex in self.vertices[:3]:
            self.facet.add_vertex(vertex)
        target = np.linalg.norm(np.cross(self.vertices[2] - self.vertices[0],
                                         self.vertices[1] - self.vertices[0])) / 2
        target *= self.facet.distance_from_origin / 3
        self.assertAlmostEqual(self.facet.get_volume(), target)

    def test_perimeter_length(self):
        """Tests perimeter length function"""
        for vertex in self.vertices[:3]:
            self.facet.add_vertex(vertex)
        self.assertAlmostEqual(self.facet.perimeter_length,
                               np.sqrt(5) + np.sqrt(9 + 4.5**2) + np.sqrt(13 + 3.5**2))
