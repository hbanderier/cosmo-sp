#!/bin/bash
module load daint-gpu
module load CDO

outputpath="/scratch/snx3000/hbanderi/cosmo-sandbox/2_lm_c"
# outputpath="/scratch/snx3000/zemanc/cosmo-setups/vericlim/2_lm_c"

# Start a new log
thisdir=`pwd`
echo "Merge log from $(date)" > $thisdir/merge_output_12h.log

# Constantly check for merges
while :
do
    # Check directories
    dirs=`find $outputpath -mindepth 3 -maxdepth 3 -type d`

    # Loop through all directories
    for d in $dirs; do
		a=$(basename $d)
		if [[ $a == "12h" ]]; then #"12h" "12h_plev" "24h" "24h_plev" "24h_100zlev"
	    	cd $d
        	# Check for a constants file and rename it if it's there
        	if test -f *"c.nc"; then
            	cfile=`ls *"c.nc"`
				mv $cfile constants.nc
        	fi
        	# Loop through years/months
        	for y in $(seq -f "%04g" 1989 1998); do
           		for m in $(seq -f "%02g" 1 12); do
                	# Check if the files haven't been merged yet
                	if ! ls lffdm${y}${m}* 1> /dev/null 2>&1; then
                    	# Check if all files for a month are there
		    			b="${a:0:2}"
		    			if [[ $b == "12" ]]; then
							datenextmonth=`date --date="${y}${m}01 + 1 month - 12 hours" +"%Y%m%d%H%M%S"`
						else
							datenextmonth=`date --date="${y}${m}01 + 1 month - 1 day" +"%Y%m%d%H%M%S"`
						fi
						echo "$d, $b, $datenextmonth"
						if compgen -G "lffd${datenextmonth}*" > /dev/null; then
                        	# Merge
                        	timemerge=`date +"%Y%m%d %H:%M:%S"`
                        	timefile=$(date --date="`stat -c "%y" lffd${datenextmonth}*`" +"%Y%m%d %H:%M:%S")
                        	echo "$d: $y$m File: $timefile Merge: $timemerge" >> $thisdir/merge_output_12h.log
                        	cdo -mergetime "lffd$y$m"* "lffdm$y$m.nc"
                        	# Delete daily/hourly files if merge was successful
                        	if [ $? -eq 0 ]; then
                            	rm lffd${y}${m}*
                        	fi
                    	fi
                	fi
            	done
        	done
		fi
    done
    sleep 1
done


