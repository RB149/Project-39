import sys

from mp_api.client import MPRester
#querying user for formula input
formula = print("Input Element/Alloy abbreviation:")
args = sys.argv
formula = str(args[1])
formula_to_retrieve=formula

#initialise interface to Materials Project (not need to include your USER_API_KEY from MP website in .pmgrc.yaml[pymatgen])
mpr = MPRester()
#api key kIviblPxcYc160uJMH3NMm1D3LYfY5Zs
#if .pmgrc.yaml not working:
#mpr = MPRester(api_key=kIviblPxcYc160uJMH3NMm1D3LYfY5Zs)

#search MP for most stable material with given formula
docs = mpr.materials.summary.search(
        energy_above_hull=(0, 0), formula = formula_to_retrieve, fields=["material_id"]
    )
stable_mpids = [doc.material_id for doc in docs]

#checking mpid used
print (stable_mpids)

#get corresponding structure
structure = mpr.get_structure_by_material_id(stable_mpids[0])

#write out structure as poscar (can visualise easily in VESTA https://jp-minerals.org/vesta/en/)
structure.get_sorted_structure().to("structure.vasp", "poscar")
