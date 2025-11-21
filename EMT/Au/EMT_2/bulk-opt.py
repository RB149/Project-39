from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.io import read, write



struct = read("./structureAg.vasp")
struct.calc=EMT()
cell = ExpCellFilter(struct)
geom_opt = BFGS(cell)
geom_opt.run(fmax=0.0001)
write('opt.vasp', struct, format="vasp")
print("Optimized energy (eV):", struct.get_potential_energy())
