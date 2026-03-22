#!/bin/bash
#querying where output files are moved to
read -p "Where to put the files?: " d_name
#moving output and source data files
mkdir $d_name
mv slab-* out* $d_name

