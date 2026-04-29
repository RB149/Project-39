from janus_core.calculations.geom_opt import GeomOpt
from ase.io import read, write
import time # finding duration of program
start_time = time.time()

struct = read("./atoms.xyz")

geom_opt = GeomOpt(
    arch="mace_mp",
    model="/scratch/qtl506/Project-39/MLIP/mace_models/2023-12-03-mace-128-L1_epoch-199.model",
    struct=struct,
    fmax=0.001,
    filter_class=None,
)
geom_opt.run()

write('opt.xyz', geom_opt.struct, format="xyz")
print("Optimized energy (eV):", struct.get_potential_energy())

print("--- %s seconds ---" % (time.time() - start_time))
