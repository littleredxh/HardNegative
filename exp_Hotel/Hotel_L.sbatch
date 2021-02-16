#!/bin/bash
#SBATCH -N 1
#SBATCH -p large-gpu
#SBATCH -t 2-00:00:00 
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  
#SBATCH --mail-user=xuanhong@gwu.edu

srun python main.py --Data ${DATA} --i ${IT} --model ${MODEL} --dim ${DIM} --lr ${LR} --epochs ${EP}  --ngpu ${NGPU} --bsize ${BSIZE} --repeat ${REPEAT} --lam ${LAM} --method ${METHOD}