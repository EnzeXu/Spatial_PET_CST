#!/bin/bash
#SBATCH --job-name="spatial_0.2_1.0"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=16GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/spatial_0.2_1.0-%j.o"
#SBATCH --error="jobs_oe/spatial_0.2_1.0-%j.e"
echo $(pwd) > "jobs/pwd.txt"
source venv/bin/activate
python spatial_simulation.py --n 3 --threshold 0.2 --diffusion_unit_rate 1.0
