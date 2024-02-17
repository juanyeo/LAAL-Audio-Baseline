current_time=$(date "+%Y%m%d_%H:%M:%S")
project_dir=/home/s2/juanyeo/laalaudio

python ${project_dir}/run.py \
--cfg ${project_dir}/config/SSAST_epicsounds.yaml \
NUM_GPUS 1 \
OUTPUT_DIR ${project_dir}/outputs/SSAST_EPIC_$current_time \
EPICSOUNDS.AUDIO_DATA_FILE /shared/s2/lab01/clab/EPIC_audio.hdf5 \
EPICSOUNDS.ANNOTATIONS_DIR ${project_dir}/data/epic-sounds \
TRAIN.CHECKPOINT_FILE_PATH /shared/s2/lab01/juan/weights/SSAST-Base-Patch-400.pth