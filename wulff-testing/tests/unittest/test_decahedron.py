#!/usr/bin/env python

import unittest

import numpy as np
from wulffpack import Decahedron
from wulffpack.decahedron import _get_decahedral_scale_factor
from wulffpack.core import BaseParticle
from wulffpack.core.geometry import get_angle
from matplotlib.colors import to_hex
from ase.build import bulk


class TestDecahedron(unittest.TestCase):
    """Test of class Decahedron"""

    def __init__(self, *args, **kwargs):
        super(TestDecahedron, self).__init__(*args, **kwargs)
        self.natoms = 500
        self.surface_energies = {(1, 1, 1): 1.2}
        self.twin_energy = 0.1
        self.primitive_structure = bulk('Pd', a=4.0)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.particle = Decahedron(surface_energies=self.surface_energies,
                                   twin_energy=self.twin_energy,
                                   primitive_structure=self.primitive_structure,
                                   natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = Decahedron(surface_energies=self.surface_energies,
                              twin_energy=self.twin_energy)
        self.assertIsInstance(particle, Decahedron)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 4)

        # Check that the volume of the particle is OK
        target_volume = 4.0**3 * 1000 / 4
        self.assertAlmostEqual(target_volume, particle.volume)

        # Test that init with too large twin energy fails
        with self.assertRaises(ValueError) as cm:
            Decahedron(surface_energies=self.surface_energies,
                       twin_energy=1.0)
        self.assertIn('The construction expects a twin energy', str(cm.exception))

        # Test that init with non-cubic crystal fails
        with self.assertRaises(ValueError) as cm:
            Decahedron(surface_energies=self.surface_energies,
                       primitive_structure=bulk('Ti', crystalstructure='hcp'),
                       twin_energy=1.0)
        self.assertIn('cubic symmetry', str(cm.exception))

    def test_atoms(self):
        """Tests atoms property"""
        atoms = self.particle.atoms
        self.assertEqual(atoms.get_chemical_formula(), 'Pd459')

    def test_fivefold_axis_vector(self):
        """Tests fivefold axis vector property"""
        vector = self.particle.fivefold_axis_vector
        self.assertIsInstance(vector, np.ndarray)
        self.assertAlmostEqual(np.linalg.norm(vector), 1.0)
        angle_to_normal = get_angle(vector, self.particle._twin_form.facets[0].normal)
        self.assertAlmostEqual(angle_to_normal, 1.57079632679)

    def test_decahedral_scale_factor(self):
        """Tests retrieval of decahedral scale factor"""
        self.assertAlmostEqual(_get_decahedral_scale_factor(), 1.027486297)

    def test_average_surface_energy(self):
        """Tests average surface energy property"""
        self.assertAlmostEqual(self.particle.average_surface_energy, 1.2)

    def test_number_of_corners(self):
        """Tests number of corners property"""
        self.assertEqual(self.particle.number_of_corners, 27)

    def test_aspect_ratio(self):
        """Tests aspect ratio property"""
        self.assertAlmostEqual(self.particle.aspect_ratio, 0.71375623)

    def test_get_continuous_color_scheme(self):
        """Tests that retrieval of a continuous color scheme works."""
        colors = self.particle.get_continuous_color_scheme()
        adapted_ret_val = {key: to_hex(val) for key, val in colors.items()}
        target_val = {(1, 1, 1): '#2980b9'}
        self.assertDictEqual(adapted_ret_val, target_val)

    def test_get_strain_energy(self):
        """Tests strain energy property"""
        shear_modulus = 0.169
        poissons_ratio = 0.44
        strain_energy = self.particle.get_strain_energy(
            poissons_ratio=poissons_ratio, shear_modulus=shear_modulus)
        self.assertAlmostEqual(strain_energy, 0.2536508928571428)

    def test_set_natoms(self):
        """Tests setting of natoms"""
        natoms_before = self.particle.natoms
        volume_before = self.particle.volume
        self.particle.natoms = 3 * natoms_before
        self.assertEqual(self.particle.natoms, 3 * natoms_before)
        self.assertAlmostEqual(self.particle.volume / volume_before, 3.0)
        self.assertEqual(self.particle.atoms.get_chemical_formula(), 'Pd1378')


class TestFacetedDecahedron(unittest.TestCase):
    """Test of base class Icosahedron with more surface energies"""

    def __init__(self, *args, **kwargs):
        super(TestFacetedDecahedron, self).__init__(*args, **kwargs)
        self.natoms = 1500
        self.surface_energies = {(1, 1, 1): 1.2,
                                 (1, 0, 0): 1.2,
                                 (1, 1, 0): 1.2,
                                 (2, 1, 0): 1.2}
        self.primitive_structure = bulk('H', crystalstructure='sc', a=1.0)
        self.twin_energy = 0.05

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.particle = Decahedron(surface_energies=self.surface_energies,
                                   primitive_structure=self.primitive_structure,
                                   twin_energy=self.twin_energy,
                                   natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = Decahedron(surface_energies=self.surface_energies,
                              primitive_structure=self.primitive_structure,
                              twin_energy=self.twin_energy,
                              natoms=self.natoms)
        self.assertIsInstance(particle, Decahedron)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 9)

        # Check that the volume of the particle is OK
        target_volume = self.primitive_structure.get_volume() * self.natoms
        self.assertAlmostEqual(target_volume, particle.volume)

    def test_atoms(self):
        """Tests atoms property"""
        atoms = self.particle.atoms
        self.assertEqual(atoms.get_chemical_formula(), 'H1374')

    def test_aspect_ratio(self):
        """Tests aspect ratio property"""
        self.assertAlmostEqual(self.particle.aspect_ratio, 0.9114231)
