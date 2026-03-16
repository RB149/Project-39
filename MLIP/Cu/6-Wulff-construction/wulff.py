from wulffpack import SingleCrystal
from ase.io import read, write

prim = read("./opt.vasp")
surface_energies = {(1, 1, 1): 0.6, (1, 0, 0): 0.8, (1, 1, 0): 0.9} #Au. same result if including the 110 index or not.
# move this file to Au folder.
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=300)
particle.view()
write('atoms.xyz', particle.atoms)


