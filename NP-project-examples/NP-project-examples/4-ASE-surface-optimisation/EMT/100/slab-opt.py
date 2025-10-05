from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import read, write

struct = read("./slab-(1, 0, 0).vasp")
struct.calc=EMT()
#note for slab opt use constant volume
geom_opt = BFGS(struct)
geom_opt.run(fmax=0.0001)
write('slab-(1, 0, 0)-opt.vasp', struct, format="vasp")
print("Optimized energy (eV):", struct.get_potential_energy())
