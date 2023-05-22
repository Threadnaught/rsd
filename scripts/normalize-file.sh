#!/bin/bash

in_file=$1
out_file=$2

min_duration_s=10

duration_s=$(ffprobe -i $in_file -show_entries format=duration -v quiet -of csv="p=0")

ffmpeg -loglevel panic -i $in_file -ar 44100 -ac 1 -map 0:a $out_file


duration_s=$(ffprobe -i $out_file -show_entries format=duration -v quiet -of csv="p=0")

if [[ -z "$duration_s" ]]; then
	echo "file failed transcode"
	exit 1
fi

if awk 'BEGIN{exit ARGV[1]<ARGV[2]}' "$min_duration_s" "$duration_s"
then
	echo "out file length ($duration_s s) < min duration ($min_duration_s s)"
	exit 1
fi

