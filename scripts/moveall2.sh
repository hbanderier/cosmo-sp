#/bin/bas!
folders=("12h_plev" "12h" "24h" "24h_plev" "24h_100zlev")
one=($(seq -f "%04g" 0 0150))
two=($(seq -f "%04g" 4100 4214))
j=0
subdir="C41"
basedir="/scratch/snx3000/hbanderi/data/${subdir}"
mkdir -p "${basedir}"
for i in $(seq 0 114); do
	if [[ -d ${two[i]} ]]; then
		dest="${basedir}/${one[$j]}"
		printf "$dest \n"
		mkdir -p "$dest"
		cd "${two[$i]}"
		mv job.out "$dest/"
		mv INPUT_* "$dest/"
		cd output
		for f in "${folders[@]}"; do
			mv -v "$f" "$dest/"
#			echo "$f, $dest"
		done
		cd ..
#	echo $"$j \r"
#	echo $"${two[$i]} \r"
#	echo $"${one[$j]} \r"
		cd ..
		j=$(( $j + 1 ))
	fi
done
