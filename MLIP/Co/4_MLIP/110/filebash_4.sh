#!/bin/bash
#querying where output files are moved to
read -p "Where to put the files?: " d_name
#moving output and source data files
mv 'slab-(1, 1, 0).vasp' 'slab-(1, 1, 0)-opt.vasp' out* $d_name
