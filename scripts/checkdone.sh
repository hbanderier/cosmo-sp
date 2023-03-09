#/bin/bash!
for i in $(seq -f "%04g" 4100 4214) ; do
	cd $i
	cd output
	cd 24h
	if [[ -f "lffdm199812.nc" ]]; then
		echo "$i is done"
	elif [[ -f "lffd19981231000000.nc" ]]; then
		echo "$i not merged"
	else
		cd ..
		cd ..
		if compgen -G "core.*" > /dev/null; then
			echo "$i crashed"
			cd ..
			rm -rvf $i
		else
			echo "$i not done"
		fi
	fi
	cd /scratch/snx3000/hbanderi/cosmo-sandbox/2_lm_c
done
