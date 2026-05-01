from wulffpack import SingleCrystal
from ase.io import read, write

prim = read("./opt.vasp")
surface_energies = {(1, 1, 1): 2.02, (1, 1, 0): 2.31, (1, 0, 0): 2.27}
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=100)
particle.view()
write('atoms.xyz', particle.atoms)


