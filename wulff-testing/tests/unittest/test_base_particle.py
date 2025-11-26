#!/usr/bin/env python

import unittest

import numpy as np
import tempfile
from ase.build import bulk
from ase import Atoms
from wulffpack.core import (BaseParticle,
                            Form,
                            Facet)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestBaseParticle(unittest.TestCase):
    """Test of base class BaseParticle"""

    def __init__(self, *args, **kwargs):
        super(TestBaseParticle, self).__init__(*args, **kwargs)
        self.prim = bulk('Pd', crystalstructure='fcc', a=4.0, cubic=True)
        self.natoms = 3000
        self.ngrains = 1
        self.energy = 1.1

        # Will form a cube
        self.reciprocal_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.symmetries = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                           np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                           np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                           np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                           np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                           np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])]

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        forms = [Form((1, 0, 0), self.energy, self.reciprocal_cell,
                      self.symmetries, (1, 0, 0))]
        self.particle = BaseParticle(forms, self.prim, self.natoms,
                                     self.ngrains)

    def test_init(self):
        """Tests that initialization of tested class works."""
        forms = [Form((1, 0, 0), self.energy, self.reciprocal_cell,
                      self.symmetries, (1, 0, 0))]
        particle = BaseParticle(forms, self.prim, self.natoms,
                                self.ngrains)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(particle.natoms, self.natoms)

        target_volume = self.prim.get_volume() / len(self.prim) * self.natoms
        self.assertAlmostEqual(particle.volume, target_volume)

        self.assertIsInstance(particle.standardized_structure, Atoms)

        self.assertEqual(len(particle.forms), 1)
        self.assertIsInstance(particle.forms[0], Form)

        # It should be a cube, simple to check areas
        for facet in particle.forms[0].facets:
            self.assertEqual(len(facet.vertices), 4)
            self.assertAlmostEqual(facet.area, target_volume ** (2 / 3.))

        # Check that the volume of the particle is OK
        target_volume = self.prim.get_volume() * self.natoms / len(self.prim)
        self.assertAlmostEqual(target_volume, particle.volume)

    def test_twin_form(self):
        """Tests twin form property"""
        self.assertEqual(self.particle._twin_form, None)
        self.particle.forms[0].parent_miller_indices = 'twin'
        self.assertIsInstance(self.particle._twin_form, Form)

    def test_make_plot(self):
        """Tests that the make_plot functionality does not break"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.particle.make_plot(ax)
        self.assertIsInstance(ax, Axes3D)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.particle.make_plot(ax, colors={(1, 0, 0): (0, 1, 0)})
        self.assertIsInstance(ax, Axes3D)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.particle.forms[0].parent_miller_indices = 'twin'
        self.particle.make_plot(ax, colors={(1, 0, 0): (0, 1, 0), 'twin': (1, 0, 0)})
        self.assertIsInstance(ax, Axes3D)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.particle.forms[0].parent_miller_indices = 'interface'
        self.particle.make_plot(ax, colors={(1, 0, 0): (0, 1, 0), 'interface': (1, 0, 0)})
        self.assertIsInstance(ax, Axes3D)

    def test_yield_facets(self):
        """Tests yield facets functionality"""
        count = 0
        for facet in self.particle._yield_facets():
            count += 1
            self.assertIsInstance(facet, Facet)
        self.assertEqual(count, 6)

    def test_area(self):
        """Tests area property"""
        target_volume = self.prim.get_volume() / len(self.prim) * self.natoms
        target_area = 6 * target_volume ** (2 / 3.)
        self.assertAlmostEqual(target_area, self.particle.area)

        # Test that twin boundaries are *not* included
        self.particle.forms[0].parent_miller_indices = 'twin'
        target_area = 0.0
        self.assertAlmostEqual(target_area, self.particle.area)

    def test_surface_energy(self):
        """Tests surface energy property"""
        target_volume = self.prim.get_volume() / len(self.prim) * self.natoms
        target_area = 6 * target_volume ** (2 / 3.)
        target_energy = target_area * self.energy
        self.assertAlmostEqual(target_energy, self.particle.surface_energy)

        # Test that twin boundaries *are* included
        self.particle.forms[0].parent_miller_indices = 'twin'
        self.assertAlmostEqual(target_energy, self.particle.surface_energy)

    def test_facet_fractions(self):
        """Tests facet fractions properties"""
        self.assertEqual(len(self.particle.facet_fractions), 1)
        self.assertAlmostEqual(self.particle.facet_fractions[(1, 0, 0)], 1.0)

        # Test that twin boundaries are *not* included
        self.particle.forms[0].parent_miller_indices = 'twin'
        self.assertEqual(len(self.particle.facet_fractions), 0)

    def test_average_surface_energy(self):
        """Tests average surface energy property"""
        self.assertAlmostEqual(self.particle.average_surface_energy, self.energy)

    def test_edge_length(self):
        """Tests average surface energy property"""
        target_value = 3 * 4 * self.particle.volume ** (1 / 3.)
        self.assertAlmostEqual(self.particle.edge_length, target_value)

    def test_number_of_corners(self):
        """Tests number of corners property"""
        self.assertEqual(self.particle.number_of_corners, 8)

    def test_duplicate_particle(self):
        """Tests duplicate particle function"""
        symmetries = [np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])]
        self.particle._duplicate_particle(symmetries)

        # Number of forms should not change
        self.assertEqual(len(self.particle.forms), 1)

        # Number of facets should double
        self.assertEqual(len(self.particle.forms[0].facets), 12)

        # There should only be 6 original grains
        count_original_grains = 0
        for facet in self.particle._yield_facets():
            if facet.original_grain:
                count_original_grains += 1
        self.assertEqual(count_original_grains, 6)

    def test_translate_particle(self):
        """Test translation functionality"""
        current_vertices = []
        for facet in self.particle._yield_facets():
            for vertex in facet.vertices:
                current_vertices.append(vertex)
        translation = np.array([3., 5., 1.])
        self.particle.translate_particle(translation)
        i = 0
        for facet in self.particle._yield_facets():
            for vertex in facet.vertices:
                self.assertTrue(np.allclose(vertex, current_vertices[i] + translation))
                i += 1

    def test_rotate_particle(self):
        """Test rotation functionality"""
        R = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        current_vertices = []
        current_normals = []
        current_cell = self.particle.standardized_structure.cell.copy()
        for facet in self.particle._yield_facets():
            for vertex in facet.vertices:
                current_vertices.append(vertex)
            current_normals.append(facet.normal)
        self.particle.rotate_particle(R)
        i = 0
        for i_facet, facet in enumerate(self.particle._yield_facets()):
            for vertex in facet.vertices:
                self.assertTrue(np.allclose(vertex, np.dot(R, current_vertices[i])))
                i += 1
            self.assertTrue(np.allclose(facet.normal, np.dot(R, current_normals[i_facet])))
        self.assertTrue(np.allclose(self.particle.standardized_structure.cell,
                                    np.dot(R, current_cell)))

        # Test that rotation fails with a matrix that is not a rotation matrix
        R[0, 1] = 3.0
        with self.assertRaises(ValueError) as cm:
            self.particle.rotate_particle(R)
        self.assertIn('is not a rotation matrix', str(cm.exception))

    def test_get_atoms(self):
        """Tests _get_atoms function"""
        atoms = self.particle._get_atoms()
        self.assertEqual(len(atoms), 3429)
        self.assertEqual(atoms.get_chemical_formula(), 'Pd3429')

    def test_set_natoms(self):
        """Tests setting of natoms"""
        natoms_before = self.particle.natoms
        volume_before = self.particle.volume
        self.particle.natoms = 3 * natoms_before
        self.assertEqual(self.particle.natoms, 3 * natoms_before)
        self.assertAlmostEqual(self.particle.volume / volume_before, 3.0)
        self.assertEqual(self.particle._get_atoms().get_chemical_formula(), 'Pd9841')

    def test_write(self):
        """Tests function for writing to file"""
        # Test that exception is raised when file format is not recognized
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
        with self.assertRaises(ValueError) as cm:
            self.particle.write(temp_file.name)
        self.assertIn('File format jpg not supported', str(cm.exception))

        # Test that wavefront obj can be written
        temp_file = tempfile.NamedTemporaryFile(suffix='.obj')
        self.particle.write(temp_file.name)
        with open(temp_file.name, 'r') as f:
            data = f.readlines()
            # Some random checks
            self.assertIn('# Vertices', data[0])
            self.assertIn('v     18.17120593     18.17120593     18.17120593\n', data)
            self.assertEqual('f    21    22    23    24\n', data[-1])
            self.assertEqual('g (1, 0, 0)\n', data[-7])
            self.assertEqual(len(data), 33)
