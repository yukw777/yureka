#!/bin/bash
export file_name=$1
split_filter () { { head -n 1 $file_name; cat; } > "$FILE"; }
export -f split_filter
tail -n +2 $file_name | split --lines=$2 --filter=split_filter --additional-suffix=$4 - $3
