# Neural Seq2grid Module
Official implementation of "[Neural Sequence-to-grid Module for Learning Symbolic Rules](https://arxiv.org/abs/2101.04921)" (AAAI 2021) by Segwang Kim, Hyoungwook Nam, Joonyoung Kim, and Kyomin Jung.

This repository contains codes to build a network that follows the sequence-input grid-output structure, such as S2G-CNN, consisting of the seq2grid module and the grid decoder.
Our seq2grid module automatically segments and aligns the input sequence into a grid.
In particular, we implement the module using differentiable nested lists, which enables end-to-end training of the model without supervision for the alignment. 
Experiments with number sequence prediction problems, computer program evaluation problems, algebraic word problems, and the babi QA tasks are possible.

Run example_arithmetic_and_algorithmic.sh: Generate the algebraic word problem, train an S2G-ACNN and decode results from the S2G-ACNN.

Our implementation is based on tensor2tensor library (https://github.com/tensorflow/tensor2tensor).


## Requirements
```
tensorflow==1.13 (tested on cuda-10.0, cudnn-7.6)
python==3.6
tensor2tensor==1.13.1
tensorflow-probability==0.6.0
mesh-tensorflow==0.0.5
nltk
pandas
mathematics_dataset (https://github.com/deepmind/mathematics_dataset)
```

The followings commands are for generating data, training a model, and evaluating the model.

Refer to ``example_arithmetic_and_algorithmic.sh`` or ``example_babi.sh`` for variable settings, e.g., PROBLEM, HPARAMS_SET, MODEL and so on.

## Generate Data and Train a Model
```
CUDA_VISIBLE_DEVICES=${GPU_NUM}, t2t-trainer 
            --generate_data \
            --data_dir=${DATA_DIR} \
            --tmp_dir=${TMP_DIR} \
            --output_dir=${MODEL_DIR} \
            --t2t_usr_dir=${T2T_USR_DIR} \
            --problem=${PROBLEM} \
            --model=${MODEL} \
            --hparams_set=${HPARAMS_SET} \
            --train_steps=${TRAIN_STEPS} 
 ```
  
## Decode
```
DECODING_PY="experiments/autoregressive_decode.py"
CUDA_VISIBLE_DEVICES=${GPU_NUM} python ${DECODING_PY} 
                      --data_dir=${DATA_DIR} \
                      --problem=${PROBLEM} \
                      --model=${MODEL} \
                      --hparams_set=${HPARAMS_SET} \
                      --t2t_usr_dir=${T2T_USR_DIR} \
                      --model_dir=${MODEL_DIR} \
                      --test_shard=${TEST_SHARD} \
                      --global_steps=${TRAIN_STEPS} 
                      --decode_to_file=${DECODE_TO_FILE} 
 ```
