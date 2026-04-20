from wulffpack import SingleCrystal
from ase.io import read, write
#time of code to run
import time
start_time = time.time()

prim = read("./opt.vasp")
surface_energies = {(1, 1, 1): 2.4, (1, 1, 0): 2.0, (1, 0, 0): 2.0}
particle = SingleCrystal(surface_energies,primitive_structure=prim,natoms=1000)
particle.view()
write('atoms.xyz', particle.atoms)

#time
print("--- %s seconds ---" % (time.time() - start_time))


