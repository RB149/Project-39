from wulffpack import SingleCrystal
from ase.io import read, write

prim = read("./opt.vasp")
surface_energies = {(1, 1, 1): 1.2, (1, 0, 0): 1.5, (1, 1, 0): 1.6} 
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=300)
particle.view()
write('atoms.xyz', particle.atoms)


