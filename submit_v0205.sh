#!/bin/bash

for threshold in "0.2" "0.5" "0.8"
do
    for diffusion_unit_rate in "1.0" "10.0" "100.0" "1000.0"
    do
        sbatch jobs/spatial_${threshold}_${diffusion_unit_rate}.slurm
    done
done



