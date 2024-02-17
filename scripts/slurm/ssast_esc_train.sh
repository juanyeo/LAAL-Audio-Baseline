#!/bin/bash

#SBATCH --job-name=juan           # Submit a job named "example"
#SBATCH --nodes=1                    # Using 1 node
#SBATCH --gres=gpu:4                 # Using 1 gpu
#SBATCH --time=0-12:00:00            # 1 hour time limit
#SBATCH --mem=40000MB                # Using 10GB CPU Memory
#SBATCH --partition=P2                # Using "b" partition 
#SBATCH --cpus-per-task=4            # Using 4 maximum processor
#SBATCH --output=./S-%x.%j.out       # Make a log file

eval "$(conda shell.bash hook)"
conda activate sound

current_time=$(date "+%Y%m%d_%H:%M:%S")
project_dir=/home/s2/juanyeo/laalaudio

for((fold=1;fold<=5;fold++));
do
    echo 'now process fold'${fold}

    # exp_dir=${base_exp_dir}/fold${fold}

    train_data=${project_dir}/data/esc50/esc_train_data_${fold}.json
    test_data=${project_dir}/data/esc50/esc_eval_data_${fold}.json
    result_dir=${project_dir}/data/esc50/esc_output/fold${fold}

    srun python ${project_dir}/run.py \
    --cfg ${project_dir}/config/SSAST_esc50.yaml \
    NUM_GPUS 4 \
    TRAIN.CHECKPOINT_FILE_PATH /shared/s2/lab01/juan/weights/SSAST-Base-Patch-400.pth \
    OUTPUT_DIR ${project_dir}/outputs/SSAST_ESC_$current_time \
    ESC.TRAIN_DATA_FILE ${train_data} \
    ESC.TEST_DATA_FILE ${test_data} \
    ESC.RESULT_DIR ${result_dir}
done

python ${project_dir}/data/esc50/get_esc_result.py