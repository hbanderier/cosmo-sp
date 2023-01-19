#!/bin/bash -l
for file in *.sbatch; do 
	sbatch $file;
done
