# /bin/bash

for i in $(seq -f "%04g" 0 9); do
	cp "resampled${i}.pickle" "../data/resampled/${i}.pickle"
done
