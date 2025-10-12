#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH -J wm2-job-20251012-zephyr
#SBATCH --nodes=1
#SBATCH -c 3
#SBATCH --chdir=/lustre/home/2501110202/work/entanglement_phase_transition
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

module load julia

export JULIA_NUM_THREADS=2

julia --sysimage ~/.julia/sysimages/sys_itensors.so ./entropy_scale.jl
