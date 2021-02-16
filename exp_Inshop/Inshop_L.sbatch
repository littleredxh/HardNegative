#!/bin/bash
#SBATCH -N 1
#SBATCH -p large-gpu
#SBATCH -t 05:00:00

srun python main.py --Data ${DATA} --i ${IT} --model ${MODEL} --dim ${DIM} --lr ${LR} --epochs ${EP}  --ngpu ${NGPU} --bsize ${BSIZE} --repeat ${REPEAT} --lam ${LAM} --method ${METHOD}