#!/bin/bash
# copying file from 1-MaterialsProject-retrieve-structure directory to here
cp ../1-MaterialsProject-retrieve-structure/opt.vasp ./
#running code and making log document
python3 surface-slab-gen.py > output3.log
