#!/bin/bash
# copying file from 1-MaterialsProject-retrieve-structure directory to here
cp ../1-MaterialsProject-retrieve-structure/structure.vasp ./
#running code and making log document
python3 bulk-opt.py > output2.log
