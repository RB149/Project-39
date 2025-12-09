#!/bin/bash
# asking for directory to be copied
read -p "Directory name: " d_name
mkdir $d_name
# copying files from MLIP_3 directory to here and making folder for them
cp /users/qtl506/scratch/Project-39/MLIP/Au/MLIP_3/$d_name/'slab-(1, 1, 0).vasp' /users/qtl506/scratch/Project-39/MLIP/Au/MLIP_4/110 
#running slurm job
sbatch 4.job
