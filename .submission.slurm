#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 1000mb
#SBATCH --ntasks 1

# list NVIDIA cards, make clean, load the cuda module, make, and run main
#lspci -vvv |& grep "NVIDIA" |& tee slurm-lspci.rjob_out

# Run All Algorithms Once
./install/bin/Radix2Test 

# Run All Algorithms 100 Times:
# ./install/bin/Radix2Test 100 1

# Run Specified Algorithms(1,2) once
# ./install/bin/Radix2Test 1 0 3

# Profile Specific Algorithm(You can profile all algorithms in one file I think)
# now=$(date +"%Y%m%d_%H%M%S")
# for ((ii = 0; ii <= 7; ii++)); do
#     # nsys profile -o profile.$ii.fft --force-overwrite true ./install/bin/Radix2Test 1 $ii && \
#         ncu -f -o profile.$ii.$now.fft --set full ./install/bin/Radix2Test 1 $ii
# done
