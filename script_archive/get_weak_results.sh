#!/bin/bash

## Process weak scaling results
for file in *.nsys-rep;
do
    echo $file
    ./nsight-systems-linux-public-DVS/bin/nsys stats --force-export true $file | tee $file.sqlite_log
    rm "${filename_with_ext%.*}.sqlite"
done
