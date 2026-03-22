#!/bin/bash
# asking for directory to be copied
read -p "Directory name: " d_name
mkdir $d_name
# copying files from MLIP_3 directory to here and making folder for them
cp /scratch/qtl506/Project-39/MLIP/Cu/MLIP_3/$d_name/'slab-(1, 0, 0).vasp' /scratch/qtl506/Project-39/MLIP/Cu/MLIP_4/100 
#running program
python slab-opt.py > output.log

#mv fourth* /users/qtl506/scratch/Project-39/MLIP/Au/MLIP_4/100/$d_name
#mv 'slab-(1, 0, 0).vasp' /users/qtl506/scratch/Project-39/MLIP/Au/MLIP_4/100/$d_name


# copying 
# cp 4.job slab-opt.py cp /users/qtl506/scratch/Project-39/MLIP/Au/MLIP_4/100/$d_name
# cd /users/qtl506/scratch/Project-39/MLIP/Au/MLIP_4/100/$d_name

#removing files not relevant to experiment
# rm 'slab-(1, 1, 0)-opt.vasp' 'slab-(1, 1, 1)-opt.vasp'
