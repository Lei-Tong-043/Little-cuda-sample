#!/bin/bash
#SBATCH -p q_amd_gpu_nvidia_2
#SBATCH -N 1
#SBATCH -o test_cuda_%j.out
#SBATCH --gpus=1
##SBATCH --greps=gpu:1


make clean
make 

./test
# ./test_device