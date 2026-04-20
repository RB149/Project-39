from wulffpack import SingleCrystal
from ase.io import read, write

prim = read("./opt.vasp")
surface_energies = {(0, 0, 1): 1.56, (1, 0, 1): 1.12, (1, 1, 1): 0.89, (1, 1, 0): 0.75, (1, 0, 0): 1.95}
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=100)
# so all miller indecies have different colours when viewed, making diagram easier to understand
colors = {(1, 0, 1): '#E69925'}

particle.view(colors=colors)

write('atoms.xyz', particle.atoms)


