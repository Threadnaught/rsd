#!/bin/bash

max_jobs=20

if [ $# -eq 0 ]; then
	echo "USAGE:"
	echo "	$0 [directory to convert]"
	exit 1
fi

files=$(find $1 | grep "\\.mp3$")

echo "Found $(echo -e "$files" | wc -l) files. This might take a while..."

for f in $files; do
	#Wait until we have a spare job
	while [ $(jobs | wc -l) -ge $max_jobs ]; do
		sleep 0.1
	done

	{
		if ./normalize-file.sh $f $f-converted.mp3; then
			mv $f-converted.mp3 $f
			echo "converted $f"
		else
			echo "File $f failed to convert. See bad-files.txt for full list"
			echo $f > bad-files.txt
			rm $f $f-converted.mp3
		fi
	} &
done

wait
