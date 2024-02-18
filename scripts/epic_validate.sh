current_time=$(date "+%Y%m%d_%H:%M:%S")
project_dir=/home/s2/juanyeo/laalaudio

python ${project_dir}/run.py \
--cfg ${project_dir}/config/UNIREPLK_epicsounds.yaml \
NUM_GPUS 1 \
TRAIN.ENABLE False \
TEST.ENABLE True \
OUTPUT_DIR ${project_dir}/outputs/VALIDATE_UNI_$current_time \
EPICSOUNDS.AUDIO_DATA_FILE /shared/s2/lab01/clab/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR ${project_dir}/data/epic-sounds \
TEST.CHECKPOINT_FILE_PATH /home/s2/juanyeo/laalaudio/outputs/UNI_EPIC_20240218_00:50:21/checkpoints/checkpoint_best.pyth