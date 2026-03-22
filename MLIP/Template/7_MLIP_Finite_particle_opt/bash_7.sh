#!/bin/bash
# copying files from 6_Wulff directory to here 
cp ../6-Wulff-construction/opt.vasp ../6-Wulff-construction/atoms.xyz ./
#running code and making log document
python3 bulk-opt.py > output7.log
