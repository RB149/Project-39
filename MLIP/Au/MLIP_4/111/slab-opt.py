from janus_core.calculations.geom_opt import GeomOpt
from ase.io import read, write

struct = read("./slab-(1, 1, 1).vasp")

geom_opt = GeomOpt(
    arch="mace_mp",
    #model path leads to  MLIP models directory
    model_path="/scratch/qtl506/Project-39/MLIP/mace_models/2023-12-03-mace-128-L1_epoch-199.model",
    struct=struct,
    fmax=0.001,
    filter_kwargs={"constant_volume" : True}
    )
#note constant volume for slab opt
geom_opt.run()

write('slab-(1, 1, 1)-opt.vasp', geom_opt.struct, format="vasp")
print("Optimized energy (eV):", struct.get_potential_energy())
