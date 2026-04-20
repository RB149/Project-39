#!/bin/bash
#querying where output files are moved to
read -p "Where to put the files?: " d_name
#moving output and source data files
mv slab-*.vasp out* $d_name
