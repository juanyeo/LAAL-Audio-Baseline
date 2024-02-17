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

current_time=$(date "+%Y%m%d_%H:%M:%S")
project_dir=/home/s2/juanyeo/laalaudio

srun python ${project_dir}/run.py \
--cfg ${project_dir}/config/UNIREPLK_epicsounds.yaml \
NUM_GPUS 1 \
OUTPUT_DIR ${project_dir}/outputs/UNI_EPIC_$current_time \
EPICSOUNDS.AUDIO_DATA_FILE /shared/s2/lab01/clab/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR ${project_dir}/data/epic-sounds 