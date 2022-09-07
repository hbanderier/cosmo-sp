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

Using the script lump.sh (don't forget to module daint CDO before)

### Custom hierarchy for better parallelism



## Script

### Where they fetch stuff
### What they do
### Where they output stuff
