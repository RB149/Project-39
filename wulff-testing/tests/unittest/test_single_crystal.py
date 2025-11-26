#!/usr/bin/env python

import pytest
import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk
from wulffpack import SingleCrystal
from wulffpack.core import BaseParticle
from matplotlib.colors import to_hex


class TestSingleCrystal(unittest.TestCase):
    """Test of class SingleCrystal"""

    def __init__(self, *args, **kwargs):
        super(TestSingleCrystal, self).__init__(*args, **kwargs)
        self.natoms = 230
        self.surface_energies = {(1, 1, 1): 1.0,
                                 (1, 0, 0): 2 / np.sqrt(3),  # should lead to RTO
                                 (2, 1, 1): 1.7}
        self.chemical_symbol = 'Pd'
        self.prim = bulk(self.chemical_symbol,
                         crystalstructure='fcc',
                         a=3.9)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.particle = SingleCrystal(surface_energies=self.surface_energies,
                                      primitive_structure=self.prim,
                                      natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = SingleCrystal(surface_energies=self.surface_energies,
                                 primitive_structure=self.prim,
                                 natoms=self.natoms)
        self.assertIsInstance(particle, SingleCrystal)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 2)

        fractions = particle.facet_fractions
        self.assertAlmostEqual(fractions[(1, 1, 1)], 0.77599076)
        self.assertAlmostEqual(fractions[(1, 0, 0)], 0.22400924)

        particle = SingleCrystal(surface_energies=self.surface_energies)
        self.assertIsInstance(particle, SingleCrystal)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 2)

        # Test that it fails when equivalent facets are specified
        with self.assertRaises(ValueError) as cm:
            SingleCrystal(surface_energies={(1, 0, 0): 1, (-1, 0, 0): 2})
        self.assertIn('(-1, 0, 0) are equivalent to (1, 0, 0) by symmetry', str(cm.exception))

        # Should still work if they have the same energy
        particle = SingleCrystal(surface_energies={(1, 0, 0): 1, (-1, 0, 0): 1})
        self.assertIsInstance(particle, SingleCrystal)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 1)

    def test_volume(self):
        """Tests the volume property"""
        target_volume = self.prim.get_volume() * self.natoms / len(self.prim)
        self.assertAlmostEqual(self.particle.volume, target_volume)

    def test_facet_fractions(self):
        """Tests facet fractions properties"""
        fractions = self.particle.facet_fractions
        self.assertAlmostEqual(fractions[(1, 1, 1)], 0.77599076)
        self.assertAlmostEqual(fractions[(1, 0, 0)], 0.22400924)

    def test_average_surface_energy(self):
        """Tests average surface energy property"""
        target = 0.77599076 * 1.0 + 0.22400924 * 2 / np.sqrt(3)
        self.assertAlmostEqual(self.particle.average_surface_energy, target)

    def test_number_of_corners(self):
        """Tests number of corners property"""
        self.assertEqual(self.particle.number_of_corners, 24)

    def test_atoms(self):
        """Tests atoms property"""
        atoms = self.particle.atoms
        self.assertEqual(atoms.get_chemical_formula(), '{}201'.format(self.chemical_symbol))

    def test_get_shifted_atoms(self):
        """Test atoms with shifted center"""
        self.particle.natoms = 6

        atoms = self.particle.get_shifted_atoms(center_shift=(3.9 / 2, 3.9 / 2, 3.9 / 2))
        self.assertEqual(len(atoms), 6)

        atoms = self.particle.get_shifted_atoms()
        self.assertEqual(len(atoms), 13)

    def test_get_continuous_color_scheme(self):
        """Tests that retrieval of a continuous color scheme works."""
        # Test default colors
        surface_energies = {(1, 1, 1): 1.0,
                            (1, 0, 0): 1.1,
                            (2, 1, 0): 1.05}
        particle = SingleCrystal(surface_energies=surface_energies)
        colors = particle.get_continuous_color_scheme()
        adapted_ret_val = {key: to_hex(val) for key, val in colors.items()}
        target_val = {(1, 1, 1): '#2980b9',
                      (1, 0, 0): '#ffe82c',
                      (2, 1, 0): '#ffa76d'}
        self.assertDictEqual(adapted_ret_val, target_val)

        # Test with normalize
        colors = particle.get_continuous_color_scheme(normalize=True)
        adapted_ret_val = {key: to_hex(val) for key, val in colors.items()}
        target_val = {(1, 0, 0): '#bbaa20',
                      (1, 1, 1): '#2e8fce',
                      (2, 1, 0): '#dc6c47'}
        self.assertDictEqual(adapted_ret_val, target_val)

        # Test with non-default colors
        base_colors = {(1, 1, 1): 'r'}
        colors = particle.get_continuous_color_scheme(base_colors=base_colors)
        adapted_ret_val = {key: to_hex(val) for key, val in colors.items()}
        target_val = {(1, 0, 0): '#ffe82c',
                      (1, 1, 1): '#ff0000',
                      (2, 1, 0): '#ffa76d'}
        self.assertDictEqual(adapted_ret_val, target_val)


class TestSingleCrystalHexagonal(unittest.TestCase):
    """Test of class SingleCrystal with hexagonal symmetry"""

    def __init__(self, *args, **kwargs):
        super(TestSingleCrystalHexagonal, self).__init__(*args, **kwargs)
        self.natoms = 500
        cell = [[4.71416065, 0., 0.],
                [-2.35707981, 4.08258319, 0.],
                [0., 0., 2.91354349]]
        scaled_positions = [[0, 0, 0],
                            [1 / 3, 2 / 3, 1 / 2],
                            [2 / 3, 1 / 3, 1 / 2]]
        self.prim = Atoms('Ti3',
                          cell=cell,
                          scaled_positions=scaled_positions)
        self.surface_energies = {(1, 1, -2, 1): 0.95,
                                 (2, -1, -1, 2): 1.0,
                                 (1, 1, -2, 0): 1.0,
                                 (2, 1, -3, 2): 1.0,
                                 (2, 1, -3, 1): 1.04,
                                 (0, 0, 0, 1): 1.12,
                                 (1, 0, -1, 0): 1.22}
        self.chemical_symbol = 'Ti'

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Setup before each test."""
        self.particle = SingleCrystal(surface_energies=self.surface_energies,
                                      primitive_structure=self.prim,
                                      natoms=self.natoms)

    def test_init(self):
        """Tests that initialization of tested class works."""
        particle = SingleCrystal(surface_energies=self.surface_energies,
                                 primitive_structure=self.prim,
                                 natoms=self.natoms)
        self.assertIsInstance(particle, SingleCrystal)
        self.assertIsInstance(particle, BaseParticle)
        self.assertEqual(len(particle.forms), 6)

    def test_facet_fractions(self):
        """Tests facet fractions properties"""
        fractions = self.particle.facet_fractions
        target_fractions = {(1, 1, -2, 1): 0.48352781878523066,
                            (2, -1, -1, 2): 0.1607489797080138,
                            (1, 1, -2, 0): 0.2446711827312851,
                            (2, 1, -3, 2): 0.02713873211991753,
                            (2, 1, -3, 1): 0.07985412601656637,
                            (0, 0, 0, 1): 0.004059160638986579}
        self.assertEqual(len(fractions), len(target_fractions))
        for key in target_fractions:
            self.assertAlmostEqual(fractions[key], target_fractions[key])


class TestSingleCrystalLowPrecision(unittest.TestCase):
    """
    Test of SingleCrystal when the number of significant digits
    in the input structure is low.
    """

    def __init__(self, *args, **kwargs):
        super(TestSingleCrystalLowPrecision, self).__init__(*args, **kwargs)
        self.prim = Atoms('HHe',
                          positions=[[0.749435] * 3, [2.25057] * 3],
                          cell=[3.0022612858] * 3,
                          pbc=[True] * 3)
        self.surface_energies = {(1, 0, 0): 1.0,
                                 (1, 1, 0): 1.0,
                                 (3, 3, 1): 0.5}

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_init_with_too_tight_symmetry(self):
        """
        Tests that initialization fails with standard symmetry tolerance
        TODO: improve error message in cases like this
        """
        with self.assertRaises(ValueError):
            SingleCrystal(primitive_structure=self.prim,
                          surface_energies=self.surface_energies)

    def test_init_with_loose_symmetry_tolerance(self):
        """
        Tests that initialization works when the symmetry tolerance
        is not as tight.
        """
        particle = SingleCrystal(primitive_structure=self.prim,
                                 surface_energies=self.surface_energies,
                                 symprec=1e-3)
        self.assertEqual(len(particle.standardized_structure), 2)


def test_symmetry_operations():
    # make sure default symmetry detection causes failure
    with pytest.raises(ValueError) as e:
        SingleCrystal(surface_energies={(1, 0, 0): 1, (-1, 0, 0): 1,
                                        (0, 1, 0): 1, (0, -1, 0): 1,
                                        (0, 0, 1): 2, (0, 0, -1): 2})
    assert '(0, 0, 1) are equivalent to (1, 0, 0) by symmetry' in str(e)

    # make sure that error is raised when _no_ symmetries are provided
    with pytest.raises(ValueError) as e:
        SingleCrystal(surface_energies={(1, 0, 0): 1, (-1, 0, 0): 1,
                                        (0, 1, 0): 1, (0, -1, 0): 1,
                                        (0, 0, 1): 2, (0, 0, -1): 2},
                      symmetry_operations=[])
    assert 'You need to provide at least one symmetry operation' in str(e)

    # make sure that SingleCrystal works with explicit symmetries
    particle = SingleCrystal(
        surface_energies={(1, 0, 0): 0.9, (-1, 0, 0): 1.1,
                          (0, 1, 0): 1.2, (0, -1, 0): 1.3,
                          (0, 0, 1): 2.2, (0, 0, -1): 1.8},
        symmetry_operations=[np.identity(3, dtype=int)])
    target_facet_fractions = {
        (1, 0, 0): 0.2173913043478261,
        (-1, 0, 0): 0.2173913043478261,
        (0, 1, 0): 0.1739130434782609,
        (0, -1, 0): 0.17391304347826086,
        (0, 0, 1): 0.10869565217391304,
        (0, 0, -1): 0.10869565217391304,
    }

    assert np.allclose(
        list(particle.facet_fractions.values()),
        list(target_facet_fractions.values())
    )


if __name__ == '__main__':
    unittest.main()
