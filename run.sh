#!/bin/bash

#SBATCH --job-name=example           # Submit a job named "example"
#SBATCH --nodes=1                    # Using 1 node
#SBATCH --gres=gpu:1                 # Using 1 gpu
#SBATCH --time=0-12:00:00            # 1 hour time limit
#SBATCH --mem=20000MB                # Using 10GB CPU Memory
#SBATCH --partition=laal                # Using "b" partition 
#SBATCH --cpus-per-task=4            # Using 4 maximum processor
#SBATCH --output=./S-%x.%j.out       # Make a log file

eval "$(conda shell.bash hook)"
conda activate sound

srun /home/s2/juanyeo/laalaudio/scripts/ast_esc_train.sh
