#!/bin/bash
trap 'kill $(jobs -p)' EXIT

# iterate through the each given file names,
for file in "$@"
do
	# show tails of each in background.
	tail -f $file &
done
# wait .. until CTRL+C
wait
