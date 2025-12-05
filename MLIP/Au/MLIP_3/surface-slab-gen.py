from pymatgen.core.surface import Lattice, SlabGenerator, Structure, generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

bulk = Structure.from_file(filename="./opt.vasp")
sga = SpacegroupAnalyzer(bulk)
conventional = sga.get_conventional_standard_structure()
#print(conventional)

#generate specific slab orientation
#arguments: bulk structure, Miller indices, slab thickness in A, vacuum gap in A
#slabgen = SlabGenerator(bulk, (1, 1, 1), 10, 10)
#all_slabs = slabgen.get_slabs()
#for slab in all_slabs:
#  print(slab)

#generate slabs for different orientations
#arguments: bulk structure, max Miller index, slab thickness in A, vacuum gap in A 
all_slabs = generate_all_slabs(conventional, 1, 75, 10)
for slab in all_slabs:
  filename="slab-{0}.vasp".format(slab.miller_index)
  slab = slab.get_orthogonal_c_slab().get_sorted_structure()
  slab.to(filename,"poscar")
