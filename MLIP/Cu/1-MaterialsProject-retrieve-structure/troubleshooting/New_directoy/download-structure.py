from mp_api.client import MPRester
with MPRester(api_key="kIviblPxcYcl6OuJMH3NMmlD3LYfY5Ws") as mpr:
    data = mpr.materials.search(material_ids=["mp-30"])
