#!/bin/bash
#SBATCH --job-name="spatial_simulation"
#SBATCH --partition=medium
#SBATCH --nodes=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=32GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/spatial_simulation-%j.o"
#SBATCH --error="jobs_oe/spatial_simulation-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python spatial_simulation.py --threshold 0.3
