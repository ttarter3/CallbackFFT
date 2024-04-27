#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 1000mb
#SBATCH --ntasks 1

# list NVIDIA cards, make clean, load the cuda module, make, and run main
#lspci -vvv |& grep "NVIDIA" |& tee slurm-lspci.rjob_out

# Run All Algorithms
./install/bin/Radix2Test
# Run Specified Algorithms
# ./install/bin/Radix2Test 1 2

# Profile Specific Algorithm(You can profile all algorithms in one file I think)
# nsys profile -o profile.3.fft --force-overwrite true ./install/bin/Radix2Test 3 && \
# 	ncu -f -o profile.3.fft --set full ./install/bin/Radix2Test 3
