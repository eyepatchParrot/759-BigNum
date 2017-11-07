#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=40
#SBATCH -o job_out
#SBATCH -e job_err
#SBATCH --mail-user=vansandt@wisc.edu
#SBATCH --mail-type=ALL
#SBATCH -p slurm_shortgpu
#SBATCH --gres=gpu:1
cd $SLURM_SUBMIT_DIR
module load cuda
time ./test
mv job_out hw6.o$SLURM_JOB_ID
mv job_err hw6.e$SLURM_JOB_ID
