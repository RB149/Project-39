from wulffpack import SingleCrystal
from ase.io import read, write

prim = read("./opt.vasp")
surface_energies = {(1, 1, 1): 0.67, (1, 0, 0): 0.87, (1, 1, 0): 0.88 }
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=2000) #300

print("area=", particle.area) # finding total area of particle and what fraction belongs to each facet of crystal
print(particle.facet_fractions)

particle.view()
write('atoms.xyz', particle.atoms)


