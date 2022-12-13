#!/bin/bash -l
#SBATCH --job-name="merge_cosmo_runs"
#SBATCH --account="pr133"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

srun ./m12h_plev.sh
