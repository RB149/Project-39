#!/bin/bash
# asking for directory to be copied
read -p "Directory name: " d_name
mkdir $d_name
# copying files from 3_MLIP directory to here and making folder for them
cp ../../3_MLIP/$d_name/'slab-(1, 0, 1).vasp' ./
#running code and making log document
python3 slab-opt.py > output4.log
