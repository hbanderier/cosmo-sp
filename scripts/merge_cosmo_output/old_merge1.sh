#!/bin/bash
module load daint-gpu
module load CDO

outputpath="/scratch/snx3000/hbanderi/cosmo-sandbox/2_lm_c"
# outputpath="/scratch/snx3000/zemanc/cosmo-setups/vericlim/2_lm_c"

# Start a new log
thisdir=`pwd`
echo "Merge log from $(date)" > $thisdir/merge_output.log

# Constantly check for merges
while :
do
    # Check directories
    dirs=`find $outputpath -mindepth 3 -maxdepth 3 -type d`

    # Loop through all directories
    for d in $dirs; do
        cd $d
        # Check for a constants file and rename it if it's there
        if test -f *"c.nc"; then
            cfile=`ls *"c.nc"`
            mv $cfile constants.nc
        fi
        # Loop through years/months
        for y in $(seq -f "%04g" 1989 1999); do
            for m in $(seq -f "%02g" 1 12); do
                # Check if the files haven't been merged yet
                if ! ls lffdm${y}${m}* 1> /dev/null 2>&1; then
                    # Check if all files for a month are there 
                    datenextmonth=`date --date="${y}${m}01 + 1 month - 1 day" +"%Y%m%d"`
                    if test -f "lffd${datenextmonth}000000.nc"; then
                        # Merge
                        timemerge=`date +"%Y%m%d %H:%M:%S"`
                        timefile=$(date --date="`stat -c "%y" lffd${datenextmonth}00*`" +"%Y%m%d %H:%M:%S")
                        echo "$d: $y$m File: $timefile Merge: $timemerge" >> $thisdir/merge_output.log
                        cdo -mergetime "lffd$y$m"* "lffdm$y$m.nc"
                        # Delete daily/hourly files if merge was successful
                        if [ $? -eq 0 ]; then
                            rm lffd${y}${m}*
                        fi
                    fi
                fi
            done
        done
    done

    sleep 1
done


