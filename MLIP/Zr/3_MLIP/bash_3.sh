#!/bin/bash
# copying file from 2_MLIP directory to here
cp ../2_MLIP/opt.vasp ./
#running code and making log document
python3 surface-slab-gen.py > output3.log
