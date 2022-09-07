#!/bin/bash -l
for file in job_*; do 
	sbatch $file;
done
