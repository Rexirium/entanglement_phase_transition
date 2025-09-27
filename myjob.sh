#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH -J wm2-job-20250927-zephyr
#SBATCH --nodes=1
#SBATCH -c 50
#SBATCH --chdir=/lustre/home/2501110202/work/entanglement_phase_transition
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

module load julia

export JULIA_NUM_THREADS=50

julia ./entropy_scale.jl
