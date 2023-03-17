#!/bin/bash -l
#SBATCH --job-name='decisions_sensi'
#SBATCH --output=/users/hbanderi/cosmo-sp/scripts/logs/decisions_sensi.out
#SBATCH --error=/users/hbanderi/cosmo-sp/scripts/logs/decisions_sensi.err
#SBATCH --account='pr133'
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=multithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate pers
srun -o python -u /users/hbanderi/cosmo-sp/scripts/decisions.py --freq 1D --test KS --ana sensi &>> /users/hbanderi/cosmo-sp/scripts/logs/decisions_sensi.python.out
