#!/bin/bash -l
#SBATCH --job-name='FDR_main'
#SBATCH --output=/users/hbanderi/cosmo-sp/scripts/logs/FDR_main.out
#SBATCH --error=/users/hbanderi/cosmo-sp/scripts/logs/FDR_main.err
#SBATCH --account='pr133'
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /users/hbanderi/setup_conda.sh
srun -o python -u /users/hbanderi/cosmo-sp/scripts/FDR.py --test KS --ana main &>> /users/hbanderi/cosmo-sp/scripts/logs/FDR_main.python.out
