current_time=$(date "+%Y%m%d_%H:%M:%S")
project_dir=/home/s2/juanyeo/laalaudio

for((fold=1;fold<=5;fold++));
do
    echo 'now process fold'${fold}

    # exp_dir=${base_exp_dir}/fold${fold}

    train_data=${project_dir}/data/esc50/esc_train_data_${fold}.json
    test_data=${project_dir}/data/esc50/esc_eval_data_${fold}.json
    result_dir=${project_dir}/data/esc50/esc_output/fold${fold}

    python ${project_dir}/run.py \
    --cfg ${project_dir}/config/UNIREPLK_esc50.yaml \
    NUM_GPUS 1 \
    OUTPUT_DIR ${project_dir}/outputs/UNI_ESC_$current_time \
    ESC.TRAIN_DATA_FILE ${train_data} \
    ESC.TEST_DATA_FILE ${test_data} \
    ESC.RESULT_DIR ${result_dir}
done

python ${project_dir}/data/esc50/get_esc_result.py