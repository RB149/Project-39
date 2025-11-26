from janus_core.calculations.geom_opt import GeomOpt
from ase.io import read, write

struct = read("./slab-(1, 0, 0).vasp")

geom_opt = GeomOpt(
    arch="mace_mp",
    model_path="./mace-mpa-0-medium.model",
    struct=struct,
    fmax=0.001,
    filter_kwargs={"constant_volume" : True}
    )
#note constant volume for slab opt
geom_opt.run()

write('slab-(1, 0, 0)-opt.vasp', geom_opt.struct, format="vasp")
print("Optimized energy (eV):", struct.get_potential_energy())
