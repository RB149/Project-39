#!/usr/bin/env python

import unittest

import numpy as np
from ase.build import bulk
from wulffpack import Winterbottom
from wulffpack.core import BaseParticle


class TestWinterbottom(unittest.TestCase):
    """Test of class SingleCrystal"""

    def __init__(self, *args, **kwargs):
        super(TestWinterbottom, self).__init__(*args, **kwargs)
        self.natoms = 600
        self.surface_energies = {(1, 1, 1): 1.0,
                                 (1, 0, 0): 2 / np.sqrt(3),  # should lead to RTO
                                 (2, 1, 1): 1.7}
        self.interface_direction = (-1, -1, -1)
        self.interface_energy = 0.5
        self.chemical_symbol = 'Pd'
        self.prim = bulk(self.chemical_symbol, crystalstructure='fcc', a=3.9)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.particle = Winterbottom(surface_energies=self.surface_energies,
                                     interface_direction=self.interface_direction,
                                     interface_energy=self.interface_energy,
                                     primitive_structure=self.prim,
                                     natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = Winterbottom(surface_energies=self.surface_energies,
                                interface_direction=self.interface_direction,
                                interface_energy=self.interface_energy,
                                primitive_structure=self.prim,
                                natoms=self.natoms)
        self.assertIsInstance(particle, Winterbottom)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 6)

        fractions = particle.facet_fractions
        self.assertEqual(len(fractions), 3)
        self.assertAlmostEqual(fractions[(1, 1, 1)], 0.64486065)
        self.assertAlmostEqual(fractions[(1, 0, 0)], 0.15594156)
        self.assertAlmostEqual(fractions['interface'], 0.19919779)

        # Check that the volume of the particle is OK
        target_volume = self.prim.get_volume() * self.natoms / len(self.prim)
        self.assertAlmostEqual(target_volume, particle.volume)

        particle = Winterbottom(surface_energies=self.surface_energies,
                                interface_direction=(3, 2, 1),
                                interface_energy=self.interface_energy)
        self.assertIsInstance(particle, Winterbottom)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 13)

        fractions = particle.facet_fractions
        self.assertAlmostEqual(fractions[(1, 1, 1)], 0.62751038)
        self.assertAlmostEqual(fractions[(1, 0, 0)], 0.16843563)
        self.assertAlmostEqual(fractions['interface'], 0.204053996)

        # Check that the volume of the particle is OK
        target_volume = 4.0**3 * 1000 / 4
        self.assertAlmostEqual(target_volume, particle.volume)

        # Check with negative interface energy
        particle = Winterbottom(surface_energies=self.surface_energies,
                                interface_direction=(3, 2, 1),
                                interface_energy=-self.interface_energy)
        self.assertIsInstance(particle, Winterbottom)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 9)
        self.assertAlmostEqual(particle.facet_fractions['interface'], 0.392553002)

        # Test that init with too large interface energy fails
        with self.assertRaises(ValueError) as cm:
            Winterbottom(surface_energies=self.surface_energies,
                         interface_direction=(3, 2, 1),
                         interface_energy=1.8)
        self.assertIn('The construction expects an absolute interface energy', str(cm.exception))

    def test_atoms(self):
        """Tests atoms property"""
        atoms = self.particle.atoms
        self.assertEqual(atoms.get_chemical_formula(), '{}626'.format(self.chemical_symbol))
