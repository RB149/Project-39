#!/usr/bin/env python

import unittest

import numpy as np
from wulffpack import Icosahedron
from wulffpack.icosahedron import _get_icosahedral_scale_factor
from wulffpack.core import BaseParticle
from wulffpack.core.geometry import get_angle
from ase.build import bulk


class TestIcosahedron(unittest.TestCase):
    """Test of base class Icosahedron"""

    def __init__(self, *args, **kwargs):
        super(TestIcosahedron, self).__init__(*args, **kwargs)
        self.natoms = 500
        self.surface_energies = {(1, 1, 1): 1.2}
        self.twin_energy = 1e-9  # Very small value removes reentrance surfaces
        self.primitive_structure = bulk('Au', a=4.0)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.particle = Icosahedron(surface_energies=self.surface_energies,
                                    twin_energy=self.twin_energy,
                                    primitive_structure=self.primitive_structure,
                                    natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = Icosahedron(surface_energies=self.surface_energies,
                               twin_energy=self.twin_energy)
        self.assertIsInstance(particle, Icosahedron)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 2)

        # Check that the volume of the particle is OK
        target_volume = 4.0**3 * 1000 / 4
        self.assertAlmostEqual(target_volume, particle.volume)

        # Test that init with too large twin energy fails
        with self.assertRaises(ValueError) as cm:
            Icosahedron(surface_energies=self.surface_energies,
                        twin_energy=1.0)
        self.assertIn('The construction expects a twin energy', str(cm.exception))

        # Test that init with non-cubic crystal fails
        with self.assertRaises(ValueError) as cm:
            Icosahedron(surface_energies=self.surface_energies,
                        primitive_structure=bulk('Ti', crystalstructure='hcp'),
                        twin_energy=self.twin_energy)
        self.assertIn('cubic symmetry', str(cm.exception))

        # Test that it fails when equivalent facets are specified
        with self.assertRaises(ValueError) as cm:
            Icosahedron(surface_energies={(1, 1, 1): 1, (-1, 1, 1): 1.2},
                        twin_energy=self.twin_energy)
        self.assertIn('(-1, 1, 1) are equivalent to (1, 1, 1) by symmetry', str(cm.exception))

        # Should still work if they have the same energy
        particle = Icosahedron(surface_energies={(1, 1, 1): 1, (-1, 1, 1): 1},
                               twin_energy=self.twin_energy)
        self.assertIsInstance(particle, Icosahedron)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 2)
        particle.view()

    def test_atoms(self):
        """Tests atoms property"""
        atoms = self.particle.atoms
        self.assertEqual(atoms.get_chemical_formula(), 'Au561')

    def test_get_two_fivefold_axes(self):
        """Tests function to get two fivefold axes"""
        fivefolds = self.particle._get_two_fivefold_axes()
        self.assertIsInstance(fivefolds, tuple)
        self.assertEqual(len(fivefolds), 2)
        for fivefold in fivefolds:
            self.assertIsInstance(fivefold, np.ndarray)
        self.assertAlmostEqual(get_angle(*fivefolds), 1.10714871779)

    def test_icosahedral_scale_factor(self):
        """Tests retrieval of icosahedral scale factor"""
        self.assertAlmostEqual(_get_icosahedral_scale_factor(), 1.08036303)

    def test_edge_length(self):
        """Tests retrieval of edge length"""
        edge = (self.particle.volume * 12 / (5 * (3 + np.sqrt(5))))**(1 / 3)
        target_edge_length = 20 * 3 * edge / 2
        self.assertAlmostEqual(self.particle.edge_length, target_edge_length)

    def test_area(self):
        """Tests retrieval of edge length"""
        edge = (self.particle.volume * 12 / (5 * (3 + np.sqrt(5))))**(1 / 3)
        target_area = 5 * np.sqrt(3) * edge**2
        self.assertAlmostEqual(self.particle.area, target_area)

    def test_number_of_corners(self):
        """Tests number of corners property"""
        self.assertEqual(self.particle.number_of_corners, 12)

    def test_get_strain_energy(self):
        """Tests strain energy property"""
        shear_modulus = 0.169
        poissons_ratio = 0.44
        strain_energy = self.particle.get_strain_energy(
            poissons_ratio=poissons_ratio, shear_modulus=shear_modulus)
        self.assertAlmostEqual(strain_energy, 2.9220582857142823)

    def test_set_natoms(self):
        """Tests setting of natoms"""
        natoms_before = self.particle.natoms
        volume_before = self.particle.volume
        self.particle.natoms = 3 * natoms_before
        self.assertEqual(self.particle.natoms, 3 * natoms_before)
        self.assertAlmostEqual(self.particle.volume / volume_before, 3.0)
        self.assertEqual(self.particle.atoms.get_chemical_formula(), 'Au1415')


class TestFacetedIcosahedron(unittest.TestCase):
    """Test of base class Icosahedron with more surface energies"""

    def __init__(self, *args, **kwargs):
        super(TestFacetedIcosahedron, self).__init__(*args, **kwargs)
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
        self.particle = Icosahedron(surface_energies=self.surface_energies,
                                    twin_energy=self.twin_energy,
                                    primitive_structure=self.primitive_structure,
                                    natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = Icosahedron(surface_energies=self.surface_energies,
                               twin_energy=self.twin_energy,
                               primitive_structure=self.primitive_structure,
                               natoms=self.natoms)
        self.assertIsInstance(particle, Icosahedron)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 4)

        # Check that the volume of the particle is OK
        target_volume = self.primitive_structure.get_volume() * self.natoms
        self.assertAlmostEqual(target_volume, particle.volume)

    def test_atoms(self):
        """Tests atoms property"""
        atoms = self.particle.atoms
        self.assertEqual(atoms.get_chemical_formula(), 'H1189')

    def test_number_of_corners(self):
        """Tests number of corners property"""
        self.assertEqual(self.particle.number_of_corners, 252)


if __name__ == '__main__':
    unittest.main()
