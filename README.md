# cosmo-sp

## Data structure

### Out of Cosmo :

The INPUT_IO file is added in this repo for clarity. The data used in this work consists in seven ensembles of 20 members each. For each ensemble member, the data is output
by COSMO two files every 12h plus 3 files every 24h, each corresponding to many variables, summed up this way :
- control (ensemble)
  - 0000 (member)
    - 12h (output folder)
      - lffd19890101000000p.nc contains variable 'FI' at several p levels for the first 12-hour span of january 1989
      - lffd19890101120000p.nc
      - ...
    - 12h_2D
      - lffd19890101000000.nc contains variable 'PS' at the surface for the first 12-hour span of january 1989
      - lffd19890101120000.nc
      - ...
    - 24h
      - lffd19890101000000p.nc contains variables 'U', 'V', 'T', 'QV' at several p levels the first day of january 1989
      - lffd19890102000000p.nc
      - ...
    - 24h_2D
      - lffd19890101000000.nc contains variables
      'T_2M', 'TD_2M', 'TMAX_2M', 'TMIN_2M', 'PMSL', 'QV_2M', 'RELHUM_2M', 'DURSUN', 'HPBL', 'AEVAP_S', 'ALHFL_S', 'ASHFL_S', 'ASOD_T', 'ASWD_S', 'ASOB_T', 'ASOB_S', 'ASWDIFU_S', 'ATHB_S', 'ATHB_T', 'AUMFL_S', 'AVMFL_S', 'CAPE_ML', 'CIN_ML', 'CLCH', 'CLCM', 'CLCL', 'CLCT', 'TQI', 'TQV', 'T_S', 'T_SO', 'SP_10M', 'VABSMX_10M', 'U_10M', 'V_10M', 'W_SO_ICE', 'W_SO', 'W_SNOW', 'TOT_PREC', 'SNOW_MELT', 'SNOW_CON', 'PREC_CON', 'FR_SNOW', 'H_SNOW', 'TOT_SNOW', 'HSURF', 'FR_LAND'
      at the surface the first day of january 1989
      - lffd19890102000000.nc
      - ...
    - 24h_100zlev
      - lffm19890101000000z.nc contains variables 'U', 'V' at z=100 m for the first day of january 1989
      - lffm19890102000000z.nc
      - ...
- ref
- sp
- diff
- diff2
- nosso

### CDO

Using the script `lump.sh` (don't forget to module daint CDO before), placed in the ensemble folder, we turn these many files (3650 or 7300 per output folder, quickly reaching the million files cap) into big monthly files (now only 120 per output folder). We then delete the originial output files using `del.sh`.
They are called `lffdm$yyyy$mm.nc`. Those are backed up in `/project/pr133/hbanderi/data_backup/${ensembleName}`

### Custom hierarchy for better parallelism

To feed the main script `one_ks.py`, we again use a different data structure to maximize IO speed / computing power balance. Each variable output by COSMO is stored in 20 files (one per semester) and has the value of this variable for all timesteps within that semester, for all ensembles, all their members and all of space, in netCDF 
DataArrays of shape (n_ensembles, n_time, nx, ny, n_memb), typically something like (7, 181, 132, 129, 20). Those files are also backed up in `/project/pr133/hbanderi/data_backup/main/` under the name `${varname}s{semester}`.

## Script

The main script `one_ks.py` performs the elementwise statistical tests using `cupy` and does so `n_sel` times.

### Where it fetches stuff

It uses the variablewise files in the `main` data folder.

### What it does

Takes most of the needed quantities from the metadata file `metadata.pkl` written by the main notebook, and takes two argument as inputs : test and freq, the test to perform ("KS", "MWU" or "T") and the oversampling frequency ("1D", "2D", "1w", "1M", etc...).

### Where they output stuff

The results folder and the frequency subfolder. Again one file per semester otherwise it's too big, plus one file for the space-averaged rejections.
Those files are backed up in `/project/pr133/hbanderi/data_backup/results/`.
