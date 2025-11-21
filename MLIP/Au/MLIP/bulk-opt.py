from janus_core.calculations.geom_opt import GeomOpt
from ase.io import read, write

struct = read("./structure.vasp")

geom_opt = GeomOpt(
    arch="mace_mp",
    model_path="./mace-mpa-0-medium.model",
    struct=struct,
    fmax=0.001,
)
geom_opt.run()

write('opt.vasp', geom_opt.struct, format="vasp")
print("Optimized energy (eV):", struct.get_potential_energy())
