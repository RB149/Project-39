from wulffpack import SingleCrystal
from ase.io import read, write

prim = read("./opt.vasp")
surface_energies = {(1, 1, 1): 2.4, (1, 1, 0): 2.0, (1, 0, 0): 2.0}
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=100)
particle.view()
write('atoms.xyz', particle.atoms)


