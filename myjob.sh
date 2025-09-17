#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH -J wm2-job-20250503-zephyr
#SBATCH --nodes=1
#SBATCH -c 50
#SBATCH --chdir=/lustre/home/2000013213/work/entanglement_phase_transition
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

source ~/.bashrc

julia --threads 50 ./entropy_scale.jl
