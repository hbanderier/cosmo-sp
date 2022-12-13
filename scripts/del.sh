#/bin/bash!
declare -a folders=("12h" "12h_2D" "24h" "24h_2D" "24h_100zlev")
for i in $(seq -f "%04g" 16 17) ; do
	cd $i
	for f in ${folders[@]}; do
		cd $f
		for y in $(seq -f "%04g" 1989 1999); do
			for m in $(seq -f "%02g" 1 12); do
				# echo `pwd -P`
				# echo `ls "lffd$y$m"* | wc -l`
				# echo "lffd$y$m"
			        if [ -f "lffdm$y$m.nc" ]; then	
					rm -v "lffd$y$m"*
				fi
			done
		done
		cd ..
	done
	cd ..
done
