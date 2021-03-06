#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_NUM=0

GPU_FRACTION=0.95
T2T_USR_DIR="./usr_dir"
DATA_DIR= # SET YOURS
TMP_DIR= # SET YOURS
SESS_DIR= # SET YOURS
DECODING_PY=".autoregressive_decode.py"
PROBLEM='algebraic_word_problem'
MODEL='seq_in_grid_out_architecture'
HPARAMS_SET='sigo_hparam_h128_l3_gru_acnn'
TRAIN_STEPS=64000
SAVE_STEPS=8000
TRIAL=4
TEST_SHARDS=("0" "1")
MODEL_DIR=${SESS_DIR}${PROBLEM}-${MODEL}-${HPARAMS_SET}.0${TRIAL}
DECODE_TO_FILE=${MODEL_DIR}/decode_00000

CUDA_VISIBLE_DEVICES=${GPU_NUM}, t2t-trainer --generate_data --data_dir=${DATA_DIR} --tmp_dir=${TMP_DIR} --output_dir=${MODEL_DIR} --t2t_usr_dir=${T2T_USR_DIR} --problem=${PROBLEM} --model=${MODEL} --hparams_set=${HPARAMS_SET} --train_steps=${TRAIN_STEPS} --local_eval_frequency=${SAVE_STEPS} 
    

for TEST_SHARD in ${TEST_SHARDS[@]};
do
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python ${DECODING_PY} --data_dir=${DATA_DIR} --problem=${PROBLEM} --model=${MODEL} --hparams_set=${HPARAMS_SET} --t2t_usr_dir=${T2T_USR_DIR} --model_dir=${MODEL_DIR} --test_shard=${TEST_SHARD} --global_steps=${TRAIN_STEPS} --gpu_fraction=${GPU_FRACTION} --decode_to_file=${DECODE_TO_FILE}
done
